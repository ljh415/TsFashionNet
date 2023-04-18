import torch.nn as nn

from utils import device
from custom_loss import LandmarkLoss, SmoothCrossEntropyLoss

class BaseLoss(nn.Module):
    def __init__(self, lm_reduction, shape_only=False, class_weight=None, smoothing=0.0):
        super(BaseLoss, self).__init__()
        
        self.device = device
        self.num_losses = 2
        self.lm_criterion = LandmarkLoss(lm_reduction).to(self.device)
        self.vis_criterion = nn.BCELoss().to(self.device)
        
        if class_weight:
            cat_weight = class_weight['category'].to(self.device)
            att_weight = class_weight['attribute'].to(self.device)
        else : cat_weight, att_weight=None, None
        
        if not shape_only:
            self.num_losses = 4
            # self.category_criterion = nn.CrossEntropyLoss(weight=cat_weight).to(self.device)
            self.category_criterion = SmoothCrossEntropyLoss(weight=cat_weight, smoothing=smoothing)
            self.attribute_criterion = nn.BCELoss(weight=att_weight).to(self.device)
    
    def forward(self, preds, targets, shape=False):
        """_summary_

        Args:
            pred (_type_): [cat_pred, att_pred, vis_pred, loc_pred]
            target (_type_): [cat_target, att_target, vis_target, loc_target]
        """
        if shape:
            vis_pred, lm_pred = preds
            vis_target, lm_target = targets
        else :
            cat_pred, att_pred, vis_pred, lm_pred = preds
            cat_target, att_target, vis_target, lm_target = targets
        
        vis_loss = self.vis_criterion(vis_pred, vis_target)
        lm_loss = self.lm_criterion(lm_pred, vis_target, lm_target)

        if shape:
            return vis_loss, lm_loss
        
        cat_target = cat_target.squeeze()
        
        cat_loss = self.category_criterion(cat_pred, cat_target)
        att_loss = self.attribute_criterion(att_pred, att_target)
        
        return cat_loss, att_loss, vis_loss, lm_loss