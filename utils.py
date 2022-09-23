import numpy as np

import torch
import torchvision.transforms as transforms

def lm_transforms(transform, landmark):
    for idx, lm in enumerate(landmark):
        lm = transforms.ToPILImage()(lm)
        lm = transform(lm)
        lm = transforms.Resize((14, 14))(lm)
        if idx == 0:
            new_lm = lm
        else :
            new_lm = torch.cat([new_lm, lm], axis=0)
    return new_lm