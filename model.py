import torch
import torch.nn as nn

class VggBackbone(nn.Module):
    def __init__(self, init_weight):
        super(VggBackbone, self).__init__()
        vgg_dict = self._make_vgg_dict(init_weight)
        self.conv1 = nn.Sequential(*vgg_dict['conv1'])
        self.conv2 = nn.Sequential(*vgg_dict['conv2'])
        self.conv3 = nn.Sequential(*vgg_dict['conv3'])
        self.conv4 = nn.Sequential(*vgg_dict['conv4'])
        self.conv5 = nn.Sequential(*vgg_dict['conv5'])
        if init_weight:
            self.conv1.apply(self._init_weight)
            self.conv2.apply(self._init_weight)
            self.conv3.apply(self._init_weight)
            self.conv4.apply(self._init_weight)
            self.conv5.apply(self._init_weight)
        else :
            self.conv5.apply(self._init_weight)
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out
    
    def _make_vgg_dict(self, shape_stream):
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True, verbose=False)
        tmp_list = []
        vgg_dict = {}
        for name, child in vgg._modules['features'].named_children():
            if not isinstance(child, nn.MaxPool2d):
                tmp_list.append(child)
            else :
                num = len(vgg_dict)+1
                if num == 5 and shape_stream:
                    pass
                else :
                    tmp_list.append(child)
                vgg_dict[f"conv{num}"] = tmp_list
                tmp_list = []
        return vgg_dict
    
    def _init_weight(self, layer):
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    
    
class TSFashionNet(nn.Module):
    def __init__(self):
        super(TSFashionNet, self).__init__()
        # texture
        self.texture_backbone = VggBackbone(init_weight=False)
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
        self.clothes_cls_fc = nn.Linear(4096, 48)
        self.attr_recog_fc = nn.Linear(4096, 1000)
        
        
        # shape
        self.shape_backbone = VggBackbone(init_weight=True)
        self.conv5_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.shape_stream = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.vis_fc = nn.Linear(256, 8)
        
        self.location = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 8, 3),
            nn.ReLU(),
        )
    
    def forward(self, x, shape=False):
        # shape
        shape_feature = self.shape_backbone(x)
        shape_out = self.shape_stream(shape_feature)
        vis_out = self.avg_pool(shape_out)
        vis_out = torch.squeeze(vis_out)
        vis_out = self.vis_fc(vis_out)
        vis_out = torch.sigmoid(vis_out)
        
        loc_out = self.location(shape_out)
        
        if shape:
            return vis_out, loc_out
        
        # texture
        texture_out = self.texture_backbone(x)
        cat_shape = self.conv5_maxpool(shape_feature).clone().detach()
        texture_out = torch.cat((texture_out, cat_shape), dim=1)
        texture_out = self.texture_stream(texture_out)
        texture_out = torch.squeeze(texture_out)
        
        clothes_out = self.clothes_cls_fc(texture_out)
        clothes_out = torch.softmax(clothes_out, dim=0)
        
        attr_out = self.attr_recog_fc(texture_out)
        attr_out = torch.sigmoid(attr_out)
        
        return vis_out, loc_out, clothes_out, attr_out