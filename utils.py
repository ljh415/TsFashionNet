import os
import cv2
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from torchmetrics import Recall, Accuracy

from square_pad import SquarePad

NORMALIZE_DICT = {
    'mean': [123.675, 116.28, 103.53],
    'std': [58.395, 57.12, 57.375]
}

device = "cuda" if torch.cuda.is_available() else "cpu"
with open('./resources/attribute_map.pickle', 'rb') as f:
    ATT_MAP = pickle.load(f)

def lm_transforms(transform, landmark):
    for idx, lm in enumerate(landmark):
        lm = transforms.ToPILImage()(lm)
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize) or isinstance(t, SquarePad):
                continue
            lm = t(lm)
        # lm = transforms.Resize((28, 28))(lm)
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

def category_check(cat_gt, cat_pred, verbose=False):
    cat_gt = cat_gt.item()
    cat_pred = torch.argmax(cat_pred).detach().cpu().numpy().item()
    if verbose:
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

def make_metric_dict():
    metric_dict = defaultdict(dict)
    metric_dict['recall'][3] = Recall(top_k=3).to(device)
    metric_dict['recall'][5] = Recall(top_k=5).to(device)
    metric_dict['acc'][3] = Accuracy(top_k=3).to(device)
    metric_dict['acc'][5] = Accuracy(top_k=5).to(device)
    
    return metric_dict

def calc_class_recall(att_gt, att_pred):

        
    def _calc_tp(k, gt_idx, pred):
        
        pred_att_dict = {idx:value for idx, value in enumerate(pred.cpu().detach().numpy())}
        pred_att_dict = dict(sorted(pred_att_dict.items(), key=lambda x : x[1], reverse=True))
        
        topk_pred = [x[0] for x in list(pred_att_dict.items())[:k]]
        topk_tp = set(topk_pred).intersection(set(gt_idx))
        
        return topk_tp
    
    att_gt_idx = [idx for idx, value in enumerate(att_gt) if value != 0]
    top3_tp = _calc_tp(3, att_gt_idx, att_pred)
    top5_tp = _calc_tp(5, att_gt_idx, att_pred)
    
    tp_dict = {
        'gt': [ATT_MAP[x] for x in att_gt_idx],
        'top3_tp': [ATT_MAP[x] for x in top3_tp],
        'top5_tp': [ATT_MAP[x] for x in top5_tp]
    }
    
    return tp_dict

def calc_metric(metric_dict, cat_pred, att_pred, cat_gt, att_gt):
    
    result_dict = {}
    
    # category
    cat_gt = cat_gt.to(device)
    cat_pred = torch.unsqueeze(cat_pred, axis=0)
    for k, metric in metric_dict['acc'].items():
        score = metric(cat_pred, cat_gt.type(torch.int16)).cpu()
        result_dict[f'category-top{k}_acc'] = score
    
    # attribute
    
    # total recall
    att_gt = att_gt.to(device)
    att_pred = torch.unsqueeze(att_pred, axis=0)
    att_gt = torch.unsqueeze(att_gt, axis=0)
    
    for k, metric in metric_dict['recall'].items():
        score = metric(att_pred, att_gt.type(torch.int16)).cpu()
        result_dict[f'attribute-top{k}_recall'] = score
    
    return result_dict