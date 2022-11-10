import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from square_pad import SquarePad

NORMALIZE_DICT = {
    'mean': [123.675, 116.28, 103.53],
    'std': [58.395, 57.12, 57.375]
}

def lm_transforms(transform, landmark):
    for idx, lm in enumerate(landmark):
        lm = transforms.ToPILImage()(lm)
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize) or isinstance(t, SquarePad):
                continue
            lm = t(lm)
        lm = transforms.Resize((28, 28))(lm)
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
    print(f"model saved : {save_path}\n")
    
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
    
def landmark_check(img, lm_out, landmark):
    height, width = img.size
    for idx, lm in enumerate(lm_out.squeeze()):
        lm = transforms.ToPILImage()(lm)
        lm = transforms.Resize((width, height))(lm)
        lm = transforms.ToTensor()(lm)
        
        if idx == 0:
            upsized_lm = lm
        else :
            upsized_lm = torch.cat([upsized_lm, lm], axis=0)
    
    lm_gt = add_weight_heatmap(img, landmark, plot=False)
    lm_pred = add_weight_heatmap(img, upsized_lm.numpy(), plot=False)
    
    plt.figure(figsize=(10, 15))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(lm_gt)
    plt.subplot(1,3,3)
    plt.imshow(lm_pred)
    plt.show()
    
    return upsized_lm

def category_check(cat_gt, cat_pred):
    cat_gt = cat_gt.item()
    cat_pred = torch.argmax(cat_pred).detach().cpu().numpy().item()
    if cat_gt == cat_pred:
        print("correct")
    else :
        print("incorrect")
    print(f"gt:\t{cat_gt}\npred:\t{cat_pred}")
    
    return cat_gt, cat_pred

def attribute_check(attr_gt, attr_pred, thr=None):
    
    attr_gt = [x[0] for x in attr_gt.nonzero().numpy()]
    
    if thr:
        attr_pred = torch.where(attr_pred>=thr, 1, 0).detach().cpu()
        attr_pred = [x[0] for x in attr_pred.nonzero().numpy()]
    else :
        attr_pred = {idx:value for idx, value in enumerate(attr_pred.detach().cpu().numpy())}
        attr_pred = dict(sorted(attr_pred.items(), reverse=True, key=lambda x: x[1]))
        attr_pred = list(attr_pred.keys())[:len(attr_gt)]
        
    # check correct
    attr_cor = [True if x in attr_pred else False for x in attr_gt]
    
    return attr_gt, attr_pred, attr_cor

def visibility_check(vis_gt, vis_pred, thr=0.8):
    vis_gt = vis_gt.numpy().astype(np.int32)
    vis_pred = torch.where(vis_pred>=thr, 1, 0).detach().cpu().numpy()
    
    return vis_gt, vis_pred, vis_gt==vis_pred