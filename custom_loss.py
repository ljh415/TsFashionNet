import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

class LandmarkLoss(nn.Module):
    def __init__(self, reduction):
        super(LandmarkLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, loc_out, visibility_batch, landmark_batch):
        batch_loss_list = []
        upsampled_loc_out = F.interpolate(loc_out, (224, 224))
        for batch_idx, visibility in enumerate(visibility_batch):
            batch_loss = 0
            for idx, v in enumerate(visibility):
                if v:
                    batch_loss += F.mse_loss(upsampled_loc_out[batch_idx][idx],landmark_batch[batch_idx, idx],
                                             reduction=self.reduction)
            batch_loss_list.append(batch_loss)
        loss = sum(batch_loss_list)/len(batch_loss_list)
        return loss


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets, train):
        if train:
            targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                self.smoothing)
        else :
            targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                0)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss