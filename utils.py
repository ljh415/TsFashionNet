import os
import cv2
import copy
import random
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchmetrics import Recall, Accuracy

NORMALIZE_DICT = {
    'mean': [123.675, 116.28, 103.53],
    'std': [58.395, 57.12, 57.375]
}

device = "cuda" if torch.cuda.is_available() else "cpu"
with open(os.path.join(os.path.expanduser("~"), 'paper', 'resources', 'attribute_map.pickle'), 'rb') as f:
    ATT_MAP = pickle.load(f)

def fix_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)

def lm_transforms(transform, landmark, flip_flag):
    for idx, lm in enumerate(landmark):
        lm = transforms.ToPILImage()(lm)
        if flip_flag:
            lm = TF.hflip(lm)
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize):
                continue
            lm = t(lm)
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
    isTensor = isinstance(img, torch.Tensor)
    if isTensor:
        # img = np.transpose(img, (1, 2, 0))
        height, width, _ = img.shape
    else :
        height, width = img.size
    check_img = np.zeros(dtype=np.float32, shape=(width, height))
    
    for i, w in enumerate(landmark):
        if isTensor:
            check_img = np.add(check_img, w)
        else :
            try:
                w = w.detach().cpu().numpy()
                check_img = np.add(check_img, w)
                
            except:
                check_img = np.add(check_img, w)
                
            # check_img += w
            
    new_h_m = np.stack([check_img*255]*3, axis=-1).astype(np.uint8)
    if isTensor:
        img *= 255/np.array(img).max()
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
    
def landmark_check(img, landmark, lm_out=None):
    print("landmark check")
    if isinstance(img, torch.Tensor):
        img = np.transpose(img, (1, 2, 0))
        height, width, _ = img.shape
    else :
        height, width = img.size
    origin_img = copy.deepcopy(img)
    
    if lm_out is not None:
        for idx, lm in enumerate(lm_out.squeeze()):
            lm = transforms.ToPILImage()(lm)
            lm = transforms.Resize((width, height))(lm)
            lm = transforms.ToTensor()(lm)
            
            if idx == 0:
                upsized_lm = lm
            else :
                upsized_lm = torch.cat([upsized_lm, lm], axis=0)
    
    lm_gt = add_weight_heatmap(img, landmark, plot=False)
    if lm_out is not None:
        lm_pred = add_weight_heatmap(img, upsized_lm, plot=False)
    
    plt.figure(figsize=(10, 15))
    plt.subplot(1,3,1)
    plt.imshow(origin_img)
    plt.subplot(1,3,2)
    plt.imshow(lm_gt)
    if lm_out is not None:
        plt.subplot(1,3,3)
        plt.imshow(lm_pred)
    plt.show()
    if lm_out is not None:
        return upsized_lm

def category_check(cat_gt, cat_pred, verbose=False):
    print("category check")
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
    print("attribute_check")
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
    metric_dict['acc'][3] = Accuracy(top_k=3, num_classes=46).to(device)
    metric_dict['acc'][5] = Accuracy(top_k=5, num_classes=46).to(device)
    
    return metric_dict

def calc_class_recall(att_gt, att_pred):
    def _calc_tp(k, gt_idx, pred):
        
        pred_att_dict = {idx:value for idx, value in enumerate(pred.cpu().detach().numpy())}
        pred_att_dict = dict(sorted(pred_att_dict.items(), key=lambda x : x[1], reverse=True))
        
        topk_pred = [x[0] for x in list(pred_att_dict.items())[:k]]
        topk_tp = set(topk_pred).intersection(set(gt_idx))
        
        return topk_tp
    
    att_gt = torch.squeeze(att_gt)
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
    
    for k, metric in metric_dict['acc'].items():
        score = metric(cat_pred, cat_gt.type(torch.int16)).cpu()
        result_dict[f'category-top{k}_acc'] = score
    
    att_pred = torch.unsqueeze(att_pred, axis=0)
    att_gt = torch.unsqueeze(att_gt, axis=0)
    
    for k, metric in metric_dict['recall'].items():
        score = metric(att_pred, att_gt.type(torch.int16)).cpu()
        result_dict[f'attribute-top{k}_recall'] = score
    
    return result_dict

def print_config(config, model, trainable=False):
    print(f"""
{'=='*20}
mode\t\t: {config['mode']}
backbone\t: {config['backbone']}
pretrained_bit\t: {config['bit_model_name']}
epochs\t\t: {config['epochs']}
shape_epochs\t: {config['shape_epochs']}
shape_lr\t: {config['shape_lr']}
init_lr\t\t: {config['lr']}
batch_size\t: {config['batch_size']}
num_of_param\t: {count_parameters(model, trainable):,}
scheduler\t: {"default" if config['scheduler'] is None else f"plateau/{config['patience']}/{config['decay_factor']}"}
loss_mode\t: {config['loss_mode']}
{'=='*20}
          """)

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else :
        return sum(p.numel() for p in model.parameters())
