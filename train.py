import os
import sys

# import wandb
import argparse
# import numpy as np
# import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import lm_resize
from model import TSFashionNet
from custom_loss import LandmarkLoss
from dataset import ShapeDataset, TSDataset

def train(args):
    
    train_path = os.path.join(args.data_path, 'train.pickle')
    valid_path = os.path.join(args.data_path, 'valid.pickle')
    
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.lr
    resolution = (args.resolution, args.resolution)
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 모른다..!
    train_transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # dataset, loader
    train_shape_dataset = ShapeDataset(train_path, transform=train_transform)
    valid_shape_dataset = ShapeDataset(valid_path)
    # train_dataset = TSDataset(train_path)
    # valid_dataset = TSDataset(valid_path)
    
    train_shape_dataloader = DataLoader(
        train_shape_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    valid_shape_dataloader = DataLoader(
        valid_shape_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )
    # valid_dataloader = DataLoader(
    #     valid_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )
    
    
    ## loss
    lm_criterion = LandmarkLoss().to(device)
    vis_criterion = nn.BCELoss().to(device)
    
    # model
    model = TSFashionNet().to(device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler?
    
    #### shape먼저 3epoch
    for epoch in range(3):
        total_loss = 0
        
        step = 1
        for batch_idx, (img_batch, landmark_batch, visibility_batch) in enumerate(train_shape_dataloader):
            
            optimizer.zero_grad()
            
            img_batch = img_batch.to(device)
            landmark_batch = landmark_batch.to(device)
            visibility_batch = visibility_batch.to(device)
            out = model(img_batch, shape=True)
            vis_out, loc_out = out

            lm_loss = lm_criterion(loc_out, visibility_batch, landmark_batch)
            vis_loss = vis_criterion(vis_out, visibility_batch)
            
            loss = lm_loss + vis_loss
            # print("loss")
            # print(loss.detach().cpu().numpy())
            
            total_loss += loss.detach().cpu().numpy()
            
            loss.backward()
            optimizer.step()
            step += 1
            
            status = (
                "\r> epoch: {:3d} > step: {:3d} > loss: {:.3f}, lr: {} ".format(
                                epoch+1,
                                step,
                                loss,
                                # total_loss/batch_size,
                                lr,
                            )
            )

            print(status, end="")
            # sys.exit(1)
            
    
    sys.exit(1)
    
    # loss
    
    
    # lr scheduler
    ## 사용 x
    
    # wandb
    
    # train
    
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nono")
    parser.add_argument("--data_path", type=str, default='/media/jaeho/SSD/datasets/deepfashion/split/')
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resolution", type=int, default=224)
    
    args = parser.parse_args()
    
    train(args)