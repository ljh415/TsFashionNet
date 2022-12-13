import os
import wandb
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Recall

from config import config, upper_class_name
from dataset import TSDataset 
from model import TSFashionNet
from square_pad import SquarePad
from custom_loss import LandmarkLoss
from utils import get_now, checkpoint_save, make_metric_dict, calc_class_recall, calc_metric

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

def train():
    
    # wandbv init ####################
    if config['wandb']:
        if config['name']:
            wandb_name = f"{config['name']}_{get_now(time=True)}"
        else :
            wandb_name = f"TSFashionNet_{get_now(time=True)}"
    
        wandb.init(entity="ljh415", project=config['project'], dir='/media/jaeho/HDD/wandb/',
                   name=wandb_name, config=config)
    
    # checkpoint save directory ######
    save_dir = os.path.join(config['ckpt_savedir'], config['project'])
        
    if config['name']:
        save_dir = os.path.join(save_dir, f"{config['name']}_{get_now(time=True)}")
    else :
        save_dir = os.path.join(save_dir, f"TSFashionNet_{get_now(time=True)}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    ######
    
    train_path = os.path.join(config['data_path'], 'train.pickle')
    valid_path = os.path.join(config['data_path'], 'valid.pickle')
    epochs = config['epochs']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    lr = config['lr']
    resolution = (config['resolution'], config['resolution'])
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    train_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(resolution),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(resolution),
        transforms.ToTensor(),
    ])
    
    # acc
    # new
    metric_dict = make_metric_dict()
    
    # prev
    # top_3_acc = Accuracy(top_k=3).to(device)
    # top_3_recall = Recall(top_k=3).to(device)
    
    # dataset, loader
    train_dataset = TSDataset(train_path, transform=train_transform, flip=config['flip'])
    valid_dataset = TSDataset(valid_path, transform=val_transform, flip=config['flip'])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    lm_criterion = LandmarkLoss().to(device)
    vis_criterion = nn.BCELoss().to(device)
    category_criterion = nn.CrossEntropyLoss().to(device)
    attribute_cretierion = nn.BCELoss().to(device)
    
    # model
    model = TSFashionNet().to(device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=6, gamma=0.1)
    
    print(f"{'='*20} training only shape {'='*20}")
    
    #### shape먼저 3epoch
    for epoch in range(3):
        model.train()
        
        step = 0
        running_shape_loss = 0
        running_landmark_loss, running_visibility_loss = 0, 0
        
        for batch_idx, (img_batch, _, _, visibility_batch, landmark_batch) in enumerate(train_dataloader):
            
            optimizer.zero_grad()
            
            img_batch = img_batch.to(device)
            landmark_batch = landmark_batch.to(device)
            visibility_batch = visibility_batch.to(device)
            
            vis_out, loc_out = model(img_batch, shape=True)  # training only shape biased stream

            # loss
            lm_loss = lm_criterion(loc_out, visibility_batch, landmark_batch)
            vis_loss = vis_criterion(vis_out, visibility_batch)
            
            loss = lm_loss + vis_loss
            
            # loss calc
            running_shape_loss += loss.detach().cpu().numpy().item()
            running_landmark_loss += lm_loss.detach().cpu().numpy().item()
            running_visibility_loss += vis_loss.detach().cpu().numpy().item()
            
            loss.backward()
            optimizer.step()
            step += 1
            
            status = (
                "\r> epoch: {:3d} > step: {:3d} > loss: {:.3f}, lr: {} ".format(
                                epoch+1,
                                step,
                                running_shape_loss/(batch_idx+1),
                                lr,
                            )
            )
            
            if args.logging_shape_train:
                if args.wandb and batch_idx % 50 == 0 :
                    wandb.log({
                        "shape_stream-train_lm_loss": running_landmark_loss/(batch_idx+1),
                        "shape_stream-train_vis_loss": running_visibility_loss/(batch_idx+1),
                        "shape_stream-train_total_loss": running_shape_loss/(batch_idx+1),
                    })

            print(status, end="")
        print()
        
        ## validate  #########
        running_val_loss = 0
        running_landmark_val_loss, running_visibility_val_loss = 0, 0
        
        with torch.no_grad():
            model.eval()
            for batch_idx, (img_batch, _, _, visibility_batch, landmark_batch) in enumerate(valid_dataloader):
                
                img_batch = img_batch.to(device)
                visibility_batch = visibility_batch.to(device)
                landmark_batch = landmark_batch.to(device)
                
                vis_out, loc_out = model(img_batch, shape=True)
                
                lm_val_loss = lm_criterion(loc_out, visibility_batch, landmark_batch)
                vis_val_loss = vis_criterion(vis_out, visibility_batch)
                
                val_loss = lm_val_loss + vis_val_loss
                
                running_val_loss += val_loss.item()
                running_landmark_val_loss += lm_val_loss.item()
                running_visibility_val_loss += vis_val_loss.item()

                if args.logging_shape_train:
                    if args.wandb and batch_idx % 50 == 0 :
                        wandb.log({
                            "shape_stream-val_lm_loss": running_landmark_val_loss/(batch_idx+1),
                            "shape_stream-val_vis_loss": running_visibility_val_loss/(batch_idx+1),
                            "shape_stream-val_total_loss": running_val_loss/(batch_idx+1),
                        })
                
        val_loss = running_val_loss / len(valid_dataloader)
        
        print("Validation loss : {:3f}\n".format(val_loss))
        
        if args.logging_shape_train:
            if epoch % args.freq_checkpoint == 0:
                checkpoint_save(model, save_dir, epoch, val_loss)
            if args.wandb:
                wandb.log({
                    "train_loss": running_shape_loss / len(train_dataloader),
                    "train_vis_loss": running_visibility_loss / len(train_dataloader),
                    "train_lm_loss": running_landmark_loss / len(train_dataloader),
                    "val_loss": val_loss,
                    "val_vis_loss": running_visibility_val_loss / len(valid_dataloader),
                    "val_lm_loss": running_landmark_val_loss / len(valid_dataloader),
                })
        
    ###### first 3 epochs, train only shape biased stream

    print(f"{'='*20} training all model {'='*20}")
    
    ## train all stream
    for epoch in range(epochs):
        model.train()
        
        step = 0
        # running_train_category_acc, running_train_attr_recall = 0, 0
        
        running_train_category_acc3, running_train_category_acc5 = 0, 0
        running_train_attr_recall3, running_train_attr_recall5 = 0, 0
        
        now_lr = lr_scheduler.optimizer.param_groups[0]['lr']
        running_train_loss = 0
        running_landmark_loss, running_visibility_loss, running_category_loss, running_attribute_loss = 0, 0, 0, 0
        
        for batch_idx, (img_batch, category_batch, attribute_batch, visibility_batch, landmark_batch) in enumerate(train_dataloader):
            
            optimizer.zero_grad()

            img_batch = img_batch.to(device)
            category_batch = category_batch.squeeze().to(device)
            attribute_batch = attribute_batch.to(device)
            visibility_batch = visibility_batch.to(device)
            landmark_batch = landmark_batch.to(device)
            
            vis_out, loc_out, category_out, attr_out = model(img_batch, shape=False)
            
            # acc
            ## new
            calc_dict = calc_metric(metric_dict, category_out, attr_out, category_batch, attribute_batch)
            running_train_category_acc3 += calc_dict['category-top3_acc'].detach().cpu().item()
            running_train_category_acc5 += calc_dict['category-top5_acc'].detach().cpu().item()
            
            running_train_attr_recall3 += calc_dict['attribute-top3_recall'].detach().cpu().item()
            running_train_attr_recall5 += calc_dict['attribute-top5_recall'].detach().cpu().item()
            
            ## prev
            # batch_category_acc = top_3_acc(category_out, category_batch)
            # running_train_category_acc += batch_category_acc.detach().cpu().item()
            
            # batch_attr_recall = top_3_recall(attr_out, attribute_batch.type(torch.int16))
            # running_train_attr_recall += batch_attr_recall.detach().cpu().item()

            # loss
            lm_loss = lm_criterion(loc_out, visibility_batch, landmark_batch)
            vis_loss = vis_criterion(vis_out, visibility_batch)
            cat_loss = category_criterion(category_out, category_batch)
            att_loss = attribute_cretierion(attr_out, attribute_batch)
            
            loss = lm_loss + vis_loss + cat_loss + 500*att_loss
            
            # loss calc
            running_train_loss += loss.detach().cpu().numpy().item()
            running_landmark_loss += lm_loss.detach().cpu().numpy().item()
            running_visibility_loss += vis_loss.detach().cpu().numpy().item()
            running_category_loss += cat_loss.detach().cpu().numpy().item()
            running_attribute_loss += att_loss.detach().cpu().numpy().item()
            
            loss.backward()
            optimizer.step()
            step += 1
            
            status = (
                "\r> epoch: {:3d} > step: {:3d} > lr: {}, loss: {:.3f}, cat acc3: {:.2f}, cat acc5: {:.2f}, attr recall3: {:.4f}, attr recall5: {:.4f}  ".format(
                                epoch+1,
                                step,
                                now_lr,
                                running_train_loss / (batch_idx+1),
                                running_train_category_acc3 / (batch_idx+1),
                                running_train_category_acc5 / (batch_idx+1),
                                running_train_attr_recall3 / (batch_idx+1),
                                running_train_attr_recall5 / (batch_idx+1)
                                
                            )
            )
            
            print(status, end="")
        print()
        
        train_loss = running_train_loss / len(train_dataloader)
        landmark_loss = running_landmark_loss / len(train_dataloader)
        visibility_loss = running_visibility_loss / len(train_dataloader)
        category_loss = running_category_loss / len(train_dataloader)
        attribute_loss = running_attribute_loss / len(train_dataloader)
        
        train_cat_acc3 = running_train_category_acc3 / len(train_dataloader)
        train_cat_acc5 = running_train_category_acc5 / len(train_dataloader)
        
        train_attr_recall3 = running_train_attr_recall3 / len(train_dataloader)
        train_attr_recall5 = running_train_attr_recall5 / len(train_dataloader)
        
        ## validate  #########
        # running_val_category_acc, running_val_attr_recall = 0, 0
        running_val_category_acc3, running_val_category_acc5 = 0, 0
        running_val_attr_recall3, running_val_attr_recall5 = 0, 0
        
        running_val_loss = 0
        running_landmark_val_loss, running_visibility_val_loss, running_category_val_loss, running_attribute_val_loss = 0, 0, 0, 0
        
        with torch.no_grad():
            model.eval()
            for img_batch, category_batch, attribute_batch, visibility_batch, landmark_batch in valid_dataloader:
                img_batch = img_batch.to(device)
                category_batch = category_batch.squeeze().to(device)
                attribute_batch = attribute_batch.to(device)
                visibility_batch = visibility_batch.to(device)
                landmark_batch = landmark_batch.to(device)
                
                vis_out, loc_out, category_out, attr_out = model(img_batch, shape=False)
                
                # acc
                ## new
                calc_dict = calc_metric(metric_dict, category_out, attr_out, category_batch, attribute_batch)
                running_val_category_acc3 += calc_dict['category-top3_acc'].detach().cpu().item()
                running_val_category_acc5 += calc_dict['category-top5_acc'].detach().cpu().item()
                
                running_val_attr_recall3 += calc_dict['attribute-top3_recall'].detach().cpu().item()
                running_val_attr_recall5 += calc_dict['attribute-top5_recall'].detach().cpu().item()
                
                ## prev
                # batch_category_acc = top_3_acc(category_out, category_batch)
                # running_val_category_acc += batch_category_acc.detach().cpu().item()
                
                # batch_attr_recall = top_3_recall(attr_out, attribute_batch.type(torch.int16))
                # running_val_attr_recall += batch_attr_recall
                
                # loss
                lm_val_loss = lm_criterion(loc_out, visibility_batch, landmark_batch)
                vis_val_loss = vis_criterion(vis_out, visibility_batch)
                cat_val_loss = category_criterion(category_out, category_batch)
                att_val_loss = attribute_cretierion(attr_out, attribute_batch)
                
                val_loss = lm_val_loss + vis_val_loss + cat_val_loss + 500*att_val_loss
                
                running_val_loss += val_loss.item()
                running_landmark_val_loss += lm_val_loss.item()
                running_visibility_val_loss += vis_val_loss.item()
                running_category_val_loss += cat_val_loss.item()
                running_attribute_val_loss += att_val_loss.item()
                
        validation_loss = running_val_loss / len(valid_dataloader)
        validation_landmark_loss = running_landmark_val_loss / len(valid_dataloader)
        validation_visibility_loss = running_visibility_val_loss / len(valid_dataloader)
        validation_category_loss = running_category_val_loss / len(valid_dataloader)
        validation_attribute_loss = running_attribute_val_loss / len(valid_dataloader)
        
        val_cat_acc3 = running_val_category_acc3 / len(valid_dataloader)
        val_cat_acc5 = running_val_category_acc5 / len(valid_dataloader)
        
        val_attr_recall3 = running_val_attr_recall3 / len(valid_dataloader)
        val_attr_recall5 = running_val_attr_recall5 / len(valid_dataloader)
         
        
        print("> Validation loss : {:3f}, cat acc3: {:2f},  cat acc5: {:2f}, attr recall3: {:4f}, attr recall5: {:4f}\n".format(
            validation_loss, 
            val_cat_acc3,
            val_cat_acc5,
            val_attr_recall3,
            val_attr_recall5
            )
        )
        
        lr_scheduler.step(epoch+1)
        
        if epoch % config['freq_checkpoint'] == 0 :
            checkpoint_save(model, save_dir, epoch, validation_loss)
        
        if config['wandb']:
            wandb_status = {
                "lr" : now_lr,
                "train_category_acc3": train_cat_acc3,
                "train_category_acc5": train_cat_acc5,
                "train_attribute_recall3": train_attr_recall3,
                "train_attribute_recall5": train_attr_recall5,
                "train_attribute": attribute_loss,
                "train_category": category_loss,
                "train_landmark": landmark_loss,
                "train_visibility": visibility_loss,
                "train_loss": train_loss,
                "valid_category_acc3": val_cat_acc3,
                "valid_category_acc5": val_cat_acc5,
                "valid_attribute_recall3": val_attr_recall3,
                "valid_attribute_recall5": val_attr_recall5,
                "valid_attribute": validation_attribute_loss,
                "valid_category": validation_category_loss,
                "valid_landmark": validation_landmark_loss,
                "valid_visibility": validation_visibility_loss,
                "valid_loss": validation_loss,
            }
            
            wandb.log(wandb_status)

def test():
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    model = TSFashionNet().to(device)
    
    ckpt_dict = torch.load(config['ckpt'])
    model.load_state_dict(ckpt_dict['model_state_dict'])
    model.eval()
    
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = TSDataset('/media/jaeho/SSD/datasets/deepfashion/split/test.pickle')

    if config['test_num'] is None:
        config['test_num'] = len(test_dataset)
        
    # set metric
    metric_dict = make_metric_dict()
    result_dict = defaultdict(lambda : defaultdict(list))
    class_recall_dict = defaultdict(lambda : defaultdict(int))
    
    # inference, calc
    for idx, data in tqdm(enumerate(test_dataset), total=config['test_num']):
        img, cat, att, _, _ = data
        img_tensor = trans(img).to(device)
        img_tensor = torch.unsqueeze(img_tensor, axis=0)
        _, _, cat_out, att_out = model(img_tensor, shape=False)
        
        # calc metric
        calc_dict = calc_metric(metric_dict, cat_out, att_out, cat, att, mode=config['mode'])
        
        for key, score in calc_dict.items():
            task, metric_name = key.split("-")
            result_dict[task][metric_name].append(score)
        
        # calc recall about classes
        tp_dict = calc_class_recall(att, att_out)
        for key, values in tp_dict.items():
            for value in values:
                class_recall_dict[key][value] += 1
        
        if idx == config['test_num']:
            break
        
    # show
    for task, score_dict in result_dict.items():
        print("=="*20)  
        print(task)
        
        for metric, score_list in score_dict.items():
            print(f"{metric}:\t{np.mean(score_list):.2f}")
        print()
    
    print("Class Recall")
    
    for upper_class, gt_counts in class_recall_dict['gt'].items():
        print("=="*30)
        print(f"{upper_class_name[upper_class]}")
        print(f"top3 : {class_recall_dict['top3_tp'][upper_class] / gt_counts}")
        print(f"top5 : {class_recall_dict['top5_tp'][upper_class] / gt_counts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--data_path", type=str, default='/media/jaeho/SSD/datasets/deepfashion/split/')
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--project", type=str, default="TSFashionNet")
    parser.add_argument("--ckpt_savedir", type=str, default="/media/jaeho/HDD/ckpt")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--freq_checkpoint", type=int, default=1)
    parser.add_argument("--logging_shape_train", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--test_num", type=int, default=None)
    parser.add_argument("--flip", action="store_true")
    
    args = parser.parse_args()
    
    config.update(vars(args))
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        if args.ckpt is None:
            raise Exception("Check the checkpoint path")
        test()