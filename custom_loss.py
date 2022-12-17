import torch.nn as nn
import torch.nn.functional as F

class LandmarkLoss(nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
    
    def forward(self, loc_out, visibility_batch, landmark_batch):
        batch_loss_list = []
        upsampled_loc_out = F.interpolate(loc_out, (224, 224))
        for batch_idx, visibility in enumerate(visibility_batch):
            batch_loss = 0
            for idx, v in enumerate(visibility):
                if v:
                    batch_loss += F.mse_loss(upsampled_loc_out[batch_idx][idx],landmark_batch[batch_idx, idx],
                                             reduction='mean')
            batch_loss_list.append(batch_loss)
        loss = sum(batch_loss_list)/len(batch_loss_list)
        return loss