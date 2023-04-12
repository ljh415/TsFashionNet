import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.layers as tml

from itertools import islice
from collections import OrderedDict

# BN => GN
# texture_stream 및 shape_stream, location을 Bit에 맞춰서 조금씩 변경
# BiT논문에서는 WS를 적용한 conv를 따로 클래스로 만들어서 사용
# 추가적으로 GroupNorm은 conv앞부분에서 적용하고, Group의 개수는 32로 통일

PreTrained_Dict = {
    '0': 'resnetv2_50x1_bitm_in21k',
    '1': 'resnetv2_50x3_bitm_in21k',
    '2': 'resnetv2_101x1_bitm_in21k',
    '3': 'resnetv2_101x3_bitm_in21k',
}

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class StdConvTransposed2d(nn.ConvTranspose2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv_transpose2d(x, w, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation) 

class GateNet(nn.Module):
    def __init__(self, channel_factor):
        super(GateNet, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(4096*channel_factor, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, a, b):
        out = torch.cat((a, b), dim=1)
        out = self.gate(out)
        return out

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
        
        self.conv11 = nn.Sequential(
            nn.Conv2d(channels, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, texture, shape):
        xi = texture + shape
        xl = self.local_att(xi)
        xg = self.global_att(xi)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * texture * wei + 2 * shape * (1-wei)
        return xo

class BiT_TSFashionNet(nn.Module):
    def __init__(self, model_name, num_of_classes=1000, bit_classifier=False):
        super(BiT_TSFashionNet, self).__init__()
        self.bit_classifier = bit_classifier
        self.texture_model = timm.create_model(model_name, pretrained=True)
        self.shape_model = timm.create_model(model_name, pretrained=False)
        self.channel_factor = 3 if 'x3' in model_name else 1
        self.gate = GateNet(self.channel_factor)
        
        self.aff = AFF(channels=2048)
        
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
        
        # adv_bit, best
        self.texture_stream = nn.Sequential(
            nn.GroupNorm(32, 2048),
            # StdConv2d(in_channels=4096, out_channels=2048, kernel_size=3, padding=0),
            nn.Conv2d(2048, 1024, 3, padding=0),
            # nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, 1024),
            # StdConv2d(in_channels=2048, out_channels=4096, kernel_size=1),
            nn.Conv2d(1024, 2048, 1),
            # nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.clothes_cls_fc = nn.Linear(2048, 46)
        self.attr_recog_fc = nn.Linear(2048, 1000)
        
        ### shape
        self.shape_backbone = nn.Sequential(OrderedDict(islice(self.shape_model._modules.items(), 2)))
        # 다초기화
        # self.shape_backbone.apply(self._init_weight)
        self.shape_stream = nn.Sequential(
            # nn.GroupNorm(32, 2048*self.channel_factor),
            # StdConv2d(in_channels=2048*self.channel_factor, out_channels=1024, kernel_size=1),
            nn.Conv2d(2048*self.channel_factor, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # nn.GroupNorm(32, 1024),
            # StdConv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.GroupNorm(32, 512),
            # StdConv2d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.Conv2d(512, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.vis_fc = nn.Linear(50176, 8)
        
        self.location = nn.Sequential(
            # nn.GroupNorm(32, 1024),
            # StdConvTransposed2d(1024, 512, 3, 2),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.GroupNorm(32, 512),
            # StdConvTransposed2d(512, 8, 3, 2),
            nn.ConvTranspose2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(1024, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        ## output = 32
        # location에 대해서 만들때 마지막 conv를 쓰는 이유는 뭐지?
        # 이 구조에 대해서 왜 이렇게 짯는지는 언급된 점이 없으니까 그냥 대충 적어도 될 것 같은데
        
        
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
        # texture_out = self.gate(texture_out, cat_shape)
        # texture_out = torch.cat([texture_out, cat_shape], dim=1)
        texture_out = self.aff(texture_out, cat_shape)
        
        texture_out = self.texture_stream(texture_out)
        texture_out = torch.squeeze(texture_out)
        
        clothes_out = self.clothes_cls_fc(texture_out)
        clothes_out = torch.softmax(clothes_out, dim=0)
        
        attr_out = self.attr_recog_fc(texture_out)
        attr_out = torch.sigmoid(attr_out)
        
        return clothes_out, attr_out, vis_out, loc_out, 