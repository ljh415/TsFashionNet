import timm
import torch
import torch.nn as nn
import timm.models.layers as tml

from itertools import islice
from collections import OrderedDict

PreTrained_Dict = {
    '0': 'resnetv2_50x1_bitm_in21k',
    '1': 'resnetv2_50x3_bitm_in21k',
    '2': 'resnetv2_101x1_bitm_in21k',
    '3': 'resnetv2_101x3_bitm_in21k',
}

class AFF(nn.Module):
    # feature fusion
    def __init__(self, channels=4096, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        # self.sigmoid = nn.Sigmoid()
        
        # exp1, 2
        # self.conv11 = nn.Sequential(
        #     nn.Conv2d(channels, 1024, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True)
        # )
        
        # exp3
        self.weight_sigmoid = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, texture, shape):
        # 원래는 x와 residual을 입력받고
        # 이 두개를 더해서 SE-Net처럼 기본의 feature를 relcalibration하는 것 같은데..
        # output shaoe of bit backbone = (batch_size, 2048, 7, 7)
        # 일단 self.conv11로 생성해서 실험 테스트 해보고... 성능 어느정도 나오는지 본 다음에 변경
        # self.texutre 다음으로 
        
        # exp1, not add texture, shape
        # xl = self.local_att(texture)
        # xg = self.global_att(shape)
        # xlg = xl+xg
        # wei = self.sigmoid(xlg)
        # xo = (2 * texture * wei) + (2 * shape * (1 - wei))
        # xo = self.conv11(xo)
        
        # exp2, add like residual
        # xi = texture + shape
        # xl = self.local_att(xi)
        # xg = self.global_att(xi)
        # xlg = xl + xg
        # wei = self.sigmoid(xlg)
        # xo = 2 * texture * wei + 2 * shape * (1-wei)
        
        # xo = self.conv11(xo)
        
        #exp3, concat input
        xi = torch.cat((texture, shape), dim=1)
        xl = self.local_att(xi)    # channel argument를 2배로 해줘야 할 것
        xg = self.global_att(xi)   # 2048 -> 4096
        xlg = xl + xg
        wei = self.weight_sigmoid(xlg)

        # wei는 4096채널이고, texture와 shape는 2048, 2048이기 때문에
        # 아래에서 에러
        # 여기에 
        xo = 2 * texture * wei + 2 * shape * (1-wei)
        xo = self.conv11(xo)
        return xo
        

class GateNet(nn.Module):
    def __init__(self, channel_factor):
        super(GateNet, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(4096*channel_factor, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
    
    def forward(self, a, b):
        out = torch.cat((a, b), dim=1)
        out = self.gate(out)
        return out

class BiT_TSFashionNet(nn.Module):
    def __init__(self, model_name, num_of_classes=1000, bit_classifier=False):
        super(BiT_TSFashionNet, self).__init__()
        self.bit_classifier = bit_classifier
        self.texture_model = timm.create_model(model_name, pretrained=True)
        self.shape_model = timm.create_model(model_name, pretrained=False)
        self.channel_factor = 3 if 'x3' in model_name else 1
        
        # prev gate
        self.gate = GateNet(self.channel_factor)
        # AFF
        self.aff = AFF()
        
        
        ### norm
        self.shape_norm = tml.GroupNormAct(2048, 32, eps=1e-5, affine=True)
        self.texture_norm = tml.GroupNormAct(2048, 32, eps=1e-5, affine=True)
        
        ### texture
        self.texture_backbone = nn.Sequential(OrderedDict(islice(self.texture_model._modules.items(), 2)))
        # 4번째 블럭 초기화
        self.texture_backbone._modules['stages']._modules['3'].apply(self._init_weight)
        # 3번째 까지는 freeze
        for key, inner_seq in islice(self.texture_backbone._modules['stages']._modules.items(), 3):
            for layer_num, layer_ in inner_seq.named_parameters():
                layer_.reauire_grad = False
        
        self.texture_stream = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 4096, 3, padding=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.clothes_cls_fc = nn.Linear(4096, 46)
        self.attr_recog_fc = nn.Linear(4096, 1000)
        
        ### shape
        self.shape_backbone = nn.Sequential(OrderedDict(islice(self.shape_model._modules.items(), 2)))
        # 다초기화
        # self.shape_backbone.apply(self._init_weight)
        self.shape_stream = nn.Sequential(
            nn.Conv2d(2048*self.channel_factor, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        self.vis_fc = nn.Linear(50176, 8)
        
        self.location = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 8, 4, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        
        ## 이건 안쓸듯..? 나중에
        # if self.bit_classifier:
        #     self.norm = tml.GroupNormAct(2048, 32, eps=1e-5, affine=True)
        #     self.head = nn.Sequential(OrderedDict([
        #         ('global_pool', tml.SelectAdaptivePool2d(pool_type='avg')),
        #         ('fc', nn.Conv2d(2048, num_of_classes, kernel_size=(1,1), strid=(1,1))),
        #         ('flatten', nn.Flatten(start_dim=1, end_dim=-1))
        #     ]))
        #     self.classifier = nn.Sequential(OrderedDict([
        #         ('norm', self.norm),
        #         ('head', self.head)
        #     ]))
            
        #     self.classifier.apply(self._init_weight)
    
    def _init_weight(self, layer):
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight)
    
    def forward(self, x, shape=False):
        ### shape
        shape_feature = self.shape_backbone(x)
        #
        shape_feature = self.shape_norm(shape_feature)

        shape_out = self.shape_stream(shape_feature)
        vis_out = torch.flatten(shape_out, start_dim=1)
        vis_out = self.vis_fc(vis_out)
        vis_out = torch.sigmoid(vis_out)
        
        loc_out = self.location(shape_out)
        
        if shape:
            return vis_out, loc_out
        
        ### texture
        texture_out = self.texture_backbone(x)
        #
        texture_out = self.texture_norm(texture_out)

        cat_shape = shape_feature.clone().detach()
        
        # prev gate
        # texture_out = self.gate(texture_out, cat_shape)
        # shape torch.Size([16, 1024, 7, 7])
        # AFF
        texture_out = self.aff(texture_out, cat_shape)
        
        texture_out = self.texture_stream(texture_out)
        texture_out = torch.squeeze(texture_out)
        
        clothes_out = self.clothes_cls_fc(texture_out)
        clothes_out = torch.softmax(clothes_out, dim=0)
        
        attr_out = self.attr_recog_fc(texture_out)
        attr_out = torch.sigmoid(attr_out)
        
        return clothes_out, attr_out, vis_out, loc_out, 