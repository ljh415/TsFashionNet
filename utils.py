import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

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
    
def add_weight_heatmap(img, landmark, alpha=0.3, plot=True):
    height, width = img.size
    check_img = np.zeros(dtype=np.float32, shape=(width, height))
    
    for w in landmark:
        check_img += w
    new_h_m = np.stack([check_img*255]*3, axis=-1).astype(np.uint8)
    origin_image = np.array(img).astype(np.uint8)
    cam_heat = cv2.applyColorMap(new_h_m, cv2.COLORMAP_JET)
    cam_heat = cv2.cvtColor(cam_heat, cv2.COLOR_RGB2BGR)
    beta = 1 - alpha
    new_img = cv2.addWeighted(cam_heat, alpha, origin_image, beta, 0)

    if plot:
        plt.imshow(new_img)
        plt.show()
    else :
        return new_img