import os
import pickle
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from utils import lm_transforms


LANDMARK_MAP = ['left_collar', 'right_collar', 'left_sleeve', 'right_sleeve',
                'left_waistline', 'right_waistline', 'left_hem', 'right_hem']

class TSDataset(Dataset):
    def __init__(self, data_path, img_dir, transform=None, flip=True):
        super(TSDataset, self).__init__()
        
        self.transform = transform
        with open(data_path, 'rb') as f:
            raw_dict = pickle.load(f)
        self.raw_data = list(raw_dict.items())
        self.flip=flip
        self.img_dir=img_dir
    
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
        
        for joint_id in range(nof_joints):
            feat_stride = np.asarray(img_size) / np.asarray(heatmap_size)
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

    def make_class_weight(self):
        def _calc_class_weight(cnt_dict):
            class_weight = [1-value/sum(cnt_dict.values()) for _, value in cnt_dict.items()]
            return torch.FloatTensor(class_weight)
        def _sort_dict(cnt_dict):
            cnt_dict = dict(sorted(cnt_dict.items(), key = lambda x:x[0]))
            return cnt_dict
        cat_cnt_dict = defaultdict(int)
        att_cnt_dict = defaultdict(int)
        for _, file_data in self.raw_data:
            for cat in file_data['category']:
                cat_cnt_dict[int(cat)] += 1
            att_infos = [idx for idx, value in enumerate(file_data['attr']) if value != -1]
            for att in att_infos:
                att_cnt_dict[int(att)] += 1
        
        cat_weight = _calc_class_weight(_sort_dict(cat_cnt_dict))
        att_weight = _calc_class_weight(_sort_dict(att_cnt_dict))
        
        return {'category': cat_weight, 'attribute': att_weight}
    
    def __getitem__(self, index):
        img_path, data_dict = self.raw_data[index]
        landmark_info = data_dict['landmark']
        
        img_path = os.path.join(self.img_dir, img_path)
        if "Striped_A-Line_Dress" in img_path:
            img_path = img_path.replace("A-Line", "A-line")
        img = Image.open(img_path).convert("RGB")
        
        landmark, visibility = self._make_landmark(img.size, landmark_info)
        category = torch.LongTensor(data_dict['category'])
        
        # 논문에서는 46개라고 했는데 확인해본 결과 총 48개의 라벨이 존재
        attribute = torch.Tensor([1 if x == 1 else 0 for x in data_dict['attr']])
        # attribute = torch.Tensor([1 if x != -1 else 0 for x in data_dict['attr']])
        
        bbox = data_dict['bbox']
        img = img.crop(bbox)
        landmark = landmark[..., bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if self.transform:
            # 이미지 transform..
            img = self.transform(img)
            if not self.flip:
                flip_flag=0
            else :
                flip_flag = np.random.randint(2)
            if flip_flag :
                img = TF.hflip(img)
                visibility = torch.tensor([visibility[1], visibility[0], visibility[3], visibility[2],
                                           visibility[5], visibility[4], visibility[7], visibility[6]])
            # landmark transform...
            landmark = lm_transforms(self.transform, landmark, flip_flag)
        
        return img, category, attribute, visibility, landmark