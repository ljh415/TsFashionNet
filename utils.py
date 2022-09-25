import os
import datetime

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

def get_now(time=False):
    now = datetime.datetime.now()
    if time:
        return now.strftime("%y%m%d-%H%M")
    return now.strftime("%y%m%d")


def checkpoint_save(model, save_dir, epoch, loss):
    save_path = os.path.join(save_dir, "checkpoint-{:03d}-{:.3f}.pth".format(epoch, loss))
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }, save_path)