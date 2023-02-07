import torch.nn as nn

from utils import device
from custom_loss import LandmarkLoss

class BaseLoss(nn.Module):
    def __init__(self, lm_reduction):
        super(BaseLoss, self).__init__()
        
        self.device = device
        self.num_losses = 4
        
        # 4개 loss
        self.lm_criterion = LandmarkLoss(lm_reduction).to(self.device)
        self.vis_criterion = nn.BCELoss().to(self.device)
        self.category_creterion = nn.CrossEntropyLoss().to(self.device)
        self.attribute_creterion = nn.BCELoss().to(self.device)
    
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
            
        # vis_target = vis_target.to(device)
        # lm_target = lm_target.to(device)
        
        vis_loss = self.vis_criterion(vis_pred, vis_target)
        lm_loss = self.lm_criterion(lm_pred, vis_target, lm_target)

        if shape:
            return vis_loss, lm_loss
        
        cat_target = cat_target.squeeze()
        
        # cat_target = cat_target.to(device)
        # att_target = att_target.to(device)
        
        cat_loss = self.category_creterion(cat_pred, cat_target)
        att_loss = self.attribute_creterion(att_pred, att_target)
        
        return cat_loss, att_loss, vis_loss, lm_loss