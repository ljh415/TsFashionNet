import torch
import torch.nn as nn

import numpy as np
import sys

class LandmarkLoss(nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
    
    def forward(self, loc_out, visibility_batch, landmark_batch):
        batch_loss_list = []
        for batch_idx, visibility in enumerate(visibility_batch):
            batch_loss = 0
            for idx, v in enumerate(visibility):
                if v:
                    ll = loc_out[batch_idx, idx] - landmark_batch[batch_idx, idx]
                    batch_loss += torch.sum(torch.norm(ll, dim=1, p=2))
            batch_loss_list.append(batch_loss)
        loss = sum(batch_loss_list)/len(batch_loss_list)
        return loss