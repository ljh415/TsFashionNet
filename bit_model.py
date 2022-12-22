import timm
import torch
import torch.nn as nn
import timm.models.layers as tml

from itertools import islice
from collections import OrderedDict

class GateNet(nn.Module):
    def __init__(self):
        super(GateNet, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=1),
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
        self.model = timm.create_model(model_name, pretrained=True)
        self.gate = GateNet()
        
        ### texture
        self.texture_backbone = nn.Sequential(OrderedDict(islice(self.model._modules.items(), 2)))
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
        self.shape_backbone = nn.Sequential(OrderedDict(islice(self.model._modules.items(), 2)))
        # 다초기화
        self.shape_backbone.apply(self._init_weight)
        self.shape_stream = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
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
        shape_out = self.shape_stream(shape_feature)
        vis_out = torch.flatten(shape_out, start_dim=1)
        vis_out = self.vis_fc(vis_out)
        vis_out = torch.sigmoid(vis_out)
        
        loc_out = self.location(shape_out)
        
        if shape:
            return vis_out, loc_out
        
        ### texture
        texture_out = self.texture_backbone(x)
        cat_shape = shape_feature.clone().detach()
        texture_out = self.gate(texture_out, cat_shape)
        texture_out = self.texture_stream(texture_out)
        texture_out = torch.squeeze(texture_out)
        
        clothes_out = self.clothes_cls_fc(texture_out)
        clothes_out = torch.softmax(clothes_out, dim=0)
        
        attr_out = self.attr_recog_fc(texture_out)
        attr_out = torch.sigmoid(attr_out)
        
        return vis_out, loc_out, clothes_out, attr_out