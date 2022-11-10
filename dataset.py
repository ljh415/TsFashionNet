import os
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import lm_transforms

# 이거 데이터셋을 굳이 두가지로 만들이유가 있을까?
# 하나의 데이터셋으로 한다음에 그냥 사용만 안하고 필요한것만 넣어서 쓰면 되는거잖아..?
# 어차피 에폭단위로 배치사이즈는 다시 도니까...


LANDMARK_MAP = ['left_collar', 'right_collar', 'left_sleeve', 'right_sleeve',
                'left_waistline', 'right_waistline', 'left_hem', 'right_hem']
IMG_DIR = '/media/jaeho/SSD/datasets/deepfashion/img-001/'

class ShapeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        
        with open(data_path, 'rb') as f:
            raw_dict = pickle.load(f)
        self.raw_data = list(raw_dict.items())
    
    def __len__(self):
        return len(self.raw_data)
    
    def _make_visibility(self, landmark_info):
        visibility = []
        for lm in LANDMARK_MAP:
            if lm in landmark_info:
                visibility.append(1 if landmark_info[lm][0] == 0 else 0)
            else:
                visibility.append(0)
        return torch.Tensor(visibility)
    
    def _make_landmark(self, img_size, landmark_info):
        height, width = img_size
        heatmap_size = img_size
        visibility = self._make_visibility(landmark_info)
        
        nof_joints = len(LANDMARK_MAP)
        joints_vis = np.array([[x] for x in visibility])
        
        target = np.zeros((nof_joints, width, height), dtype=np.float32)
        target_weight = np.ones((nof_joints, 1), dtype=np.float32)
        
        joints = [landmark_info[x][1:] if x in landmark_info else [0, 0] for x in LANDMARK_MAP]
        joints = np.array(joints, dtype=np.float32)
        
        heatmap_sigma = 3
        tmp_size = heatmap_sigma * 3
        feat_stride = np.asarray(img_size) / np.asarray(heatmap_size)
        
        for joint_id in range(nof_joints):
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue
            
            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            
            # the gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x-x0) ** 2 + (y - y0) ** 2) / (2 * heatmap_sigma ** 2))
            
            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            
            # image rnage
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]

            if v > 0.5:
                target[joint_id][img_y[0] : img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target, visibility
    
    def __getitem__(self, index):
        img_path, data_dict = self.raw_data[index]
        landmark_info = data_dict['landmark']
        
        img_path = os.path.join(IMG_DIR, img_path)
        if "Striped_A-Line_Dress" in img_path:
            img_path = img_path.replace("A-Line", "A-line")
        img = Image.open(img_path).convert("RGB")
        
        landmark, visibility = self._make_landmark(img.size, landmark_info)
        
        if self.transform:
            img = self.transform(img)
            landmark = lm_transforms(self.transform, landmark)
            
        return img, landmark, visibility

class TSDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(TSDataset, self).__init__()
        
        self.transform = transform
        with open(data_path, 'rb') as f:
            raw_dict = pickle.load(f)
        self.raw_data = list(raw_dict.items())
    
    def __len__(self):
        return len(self.raw_data)

    def _make_visibility(self, landmark_info):
        visibility = []
        for lm in LANDMARK_MAP:
            if lm in landmark_info:
                visibility.append(1 if landmark_info[lm][0] == 0 else 0)
            else :
                visibility.append(0)
        return torch.Tensor(visibility)

    def _make_landmark(self, img_size, landmark_info):
        height, width = img_size
        heatmap_size = img_size # 이걸 다르게?
        visibility = self._make_visibility(landmark_info)
        
        nof_joints = len(LANDMARK_MAP)
        joints_vis = np.array([[x] for x in visibility])
        
        target = np.zeros((nof_joints, width, height), dtype=np.float32)
        target_weight = np.ones((nof_joints, 1), dtype=np.float32)
        
        joints = [landmark_info[x][1:] if x in landmark_info else [0, 0] for x in LANDMARK_MAP]
        joints = np.array(joints, dtype=np.float32)
        
        heatmap_sigma = 3
        tmp_size = heatmap_sigma * 3
        
        for joint_id in range(nof_joints):
            feat_stride = np.asarray(img_size) / np.asarray(heatmap_size)  # 이게 같으면 늘 1일텐데
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue
            
            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            
            # the gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x-x0) ** 2 + (y - y0) ** 2) / (2 * heatmap_sigma ** 2))            
            
            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            
            # image rnage
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]

            if v > 0.5:
                target[joint_id][img_y[0] : img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target, visibility

    def __getitem__(self, index, crop=True):
        img_path, data_dict = self.raw_data[index]
        landmark_info = data_dict['landmark']
        
        img_path = os.path.join(IMG_DIR, img_path)
        if "Striped_A-Line_Dress" in img_path:
            img_path = img_path.replace("A-Line", "A-line")
        img = Image.open(img_path).convert("RGB")
        
        landmark, visibility = self._make_landmark(img.size, landmark_info)
        category = torch.LongTensor(data_dict['category'])
        
        # 논문에서는 46개라고 했는데 확인해본 결과 총 48개의 라벨이 존재
        # attribute = torch.Tensor([1 if x == 1 else 0 for x in data_dict['attr']])
        attribute = torch.Tensor([1 if x != -1 else 0 for x in data_dict['attr']])
        
        if crop:
            bbox = data_dict['bbox']
            img = img.crop(bbox)
            # img = Image.fromarray(np.array(img)[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            landmark = landmark[..., bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if self.transform:
            img = self.transform(img)
            landmark = lm_transforms(self.transform, landmark)
        
        return img, category, attribute, visibility, landmark