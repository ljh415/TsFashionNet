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

class BiT_TSFashionNet_texture(nn.Module):
    def __init__(self, model_name, num_of_classes=1000, bit_classifier=False):
        super(BiT_TSFashionNet_texture, self).__init__()
        self.bit_classifier = bit_classifier
        # self.texture_model = timm.create_model(model_name, pretrained=True)
        self.texture_backbone = timm.create_model(model_name, pretrained=True)
        
        self.channel_factor = 3 if 'x3' in model_name else 1
        self.gate = GateNet(self.channel_factor)
        
        ### norm
        # self.texture_norm = tml.GroupNormAct(2048, 32, eps=1e-5, affine=True)
        
        ### texture
        # self.texture_backbone = nn.Sequential(OrderedDict(islice(self.texture_model._modules.items(), 3)))
        # 4번째 블럭 초기화
        self.texture_backbone._modules['stages']._modules['3'].apply(self._init_weight)
        # 3번째 까지는 freeze
        for key, inner_seq in islice(self.texture_backbone._modules['stages']._modules.items(), 3):
            for layer_num, layer_ in inner_seq.named_parameters():
                layer_.reauire_grad = False
        

        self.clothes_cls_fc = nn.Linear(21843, 46)
        self.attr_recog_fc = nn.Linear(21843, 1000)
        
        
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
        ### texture
        texture_out = self.texture_backbone(x)
        # texture_out = self.texture_norm(texture_out)
        
        # texture_out = self.texture_stream(texture_out)
        # texture_out = torch.squeeze(texture_out)
        
        clothes_out = self.clothes_cls_fc(texture_out)
        clothes_out = torch.softmax(clothes_out, dim=0)
        
        attr_out = self.attr_recog_fc(texture_out)
        attr_out = torch.sigmoid(attr_out)
        
        return clothes_out, attr_out