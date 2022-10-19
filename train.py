import os
import wandb
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Recall

from dataset import TSDataset 
from model import TSFashionNet
from square_pad import SquarePad
from custom_loss import LandmarkLoss
from utils import get_now, checkpoint_save, NORMALIZE_DICT

def train(args):
    
    # wandbv init ####################
    if args.wandb:
        if args.name:
            wandb_name = f"{args.name}_{get_now(time=True)}"
        else :
            wandb_name = f"TSFashionNet_{get_now(time=True)}"
    
        wandb.init(entity="ljh415", project=args.project, dir='/media/jaeho/HDD/wandb/', name=wandb_name)
    
    # checkpoint save directory ######
    save_dir = os.path.join(args.ckpt_savedir, args.project)
        
    if args.name:
        save_dir = os.path.join(save_dir, f"{args.name}_{get_now(time=True)}")
    else :
        save_dir = os.path.join(save_dir, f"TSFashionNet_{get_now(time=True)}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    ######
    
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
    
    train_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(NORMALIZE_DICT['mean'], NORMALIZE_DICT['std'])
    ])
    val_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(NORMALIZE_DICT['mean'], NORMALIZE_DICT['std'])
    ])
    
    # acc
    top_3_acc = Accuracy(top_k=3).to(device)
    top_3_recall = Recall(top_k=3).to(device)
    
    # dataset, loader
    train_dataset = TSDataset(train_path, transform=train_transform)
    valid_dataset = TSDataset(valid_path, transform=val_transform)
    
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
    
    ## loss
    loss_dict ={
        "train" : {
            "attribute": [],
            "category": [],
            "visibility": [],
            "landmark": [],
            "train": [],
        },
        "validation" :{
            "attribute": [],
            "category": [],
            "visibility": [],
            "landmark": [],
            "validation": []
        }
    }
    
    
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
    
    # #### shape먼저 3epoch
    for epoch in range(3):
    # for epoch in range(epochs):
        model.train()
        
        # shape_loss = 0
        step = 0
        running_shape_loss = 0
        # running_landmark_loss, running_visibility_loss = 0, 0
        
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
            # running_landmark_loss += lm_loss.detach().cpu().numpy().item()
            # running_visibility_loss += vis_loss.detach().cpu().numpy().item()
            
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

            print(status, end="")
        print()
        
        shape_loss = running_shape_loss / len(train_dataloader)
        # landmark_loss = running_landmark_loss / len(train_dataloader)
        # visibility_loss = running_visibility_loss / len(train_dataloader)
        
        ## validate  #########
        running_val_loss = 0
        # running_landmark_val_loss, running_visibility_val_loss = 0, 0
        
        with torch.no_grad():
            model.eval()
            for img_batch, _, _, visibility_batch, landmark_batch in valid_dataloader:
                
                img_batch = img_batch.to(device)
                visibility_batch = visibility_batch.to(device)
                landmark_batch = landmark_batch.to(device)
                
                vis_out, loc_out = model(img_batch, shape=True)
                
                lm_val_loss = lm_criterion(loc_out, visibility_batch, landmark_batch)
                vis_val_loss = vis_criterion(vis_out, visibility_batch)
                
                val_loss = lm_val_loss + vis_val_loss
                
                running_val_loss += val_loss.item()
                # running_landmark_val_loss += lm_val_loss.item()
                # running_visibility_val_loss += vis_val_loss.item()
                
        val_loss = running_val_loss / len(valid_dataloader)
        
        print("Validation loss : {:3f}\n".format(val_loss))
        
        # if epoch % args.freq_checkpoint == 0:
        #     checkpoint_save(model, save_dir, epoch, val_loss)
        # if args.wandb:
        #     wandb.log({
        #         "train_loss": shape_loss,
        #         "train_vis_loss": visibility_loss,
        #         "train_lm_loss": landmark_loss,
        #         "val_loss": val_loss,
        #         "val_vis_loss": running_visibility_val_loss / len(valid_dataloader),
        #         "val_lm_loss": running_landmark_val_loss / len(valid_dataloader),
        #     })
    
    ###### first 3 epochs, train only shape biased stream
    
    print(f"{'='*20} training all model {'='*20}")
    
    ## train all stream
    for epoch in range(epochs):
        model.train()
        
        step = 0
        running_train_category_acc = 0
        running_train_attr_recall = 0
        
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
            batch_category_acc = top_3_acc(category_out, category_batch)
            running_train_category_acc += batch_category_acc.detach().cpu().item()
            
            batch_attr_recall = top_3_recall(attr_out, attribute_batch.type(torch.int16))
            running_train_attr_recall += batch_attr_recall.detach().cpu().item()

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
                "\r> epoch: {:3d} > step: {:3d} > lr: {}, loss: {:.3f}, cat acc: {:.2f}, attr recall: {:.4f}  ".format(
                                epoch+1,
                                step,
                                now_lr,
                                running_train_loss / (batch_idx+1),
                                running_train_category_acc / (batch_idx+1),
                                running_train_attr_recall / (batch_idx+1)
                            )
            )
            
            print(status, end="")
        print()
        
        train_loss = running_train_loss / len(train_dataloader)
        landmark_loss = running_landmark_loss / len(train_dataloader)
        visibility_loss = running_visibility_loss / len(train_dataloader)
        category_loss = running_category_loss / len(train_dataloader)
        attribute_loss = running_attribute_loss / len(train_dataloader)
        
        train_cat_acc = running_train_category_acc / len(train_dataloader)
        train_attr_recall = running_train_attr_recall / len(train_dataloader)
        
        loss_dict['train']['train'].append(train_loss)
        loss_dict['train']['landmark'].append(landmark_loss)
        loss_dict['train']['visibility'].append(visibility_loss)
        loss_dict['train']['category'].append(category_loss)
        loss_dict['train']['attribute'].append(attribute_loss)
        
        
        ## validate  #########
        running_val_category_acc = 0
        running_val_attr_recall = 0
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
                batch_category_acc = top_3_acc(category_out, category_batch)
                running_val_category_acc += batch_category_acc.detach().cpu().item()
                
                batch_attr_recall = top_3_recall(attr_out, attribute_batch.type(torch.int16))
                running_val_attr_recall += batch_attr_recall
                
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
        
        val_cat_acc = running_val_category_acc / len(valid_dataloader)
        val_attr_recall = running_val_attr_recall / len(valid_dataloader)
        
        
        print("> Validation loss : {:3f}, cat acc: {:2f}, attr recall: {:4f}\n".format(
            validation_loss, 
            val_cat_acc,
            val_attr_recall
            )
        )
        
        loss_dict['validation']['validation'].append(validation_loss)
        loss_dict['validation']['landmark'].append(validation_landmark_loss)
        loss_dict['validation']['visibility'].append(validation_visibility_loss)
        loss_dict['validation']['category'].append(validation_category_loss)
        loss_dict['validation']['attribute'].append(validation_attribute_loss)
        
        
        lr_scheduler.step(epoch+1)
        
        if epoch % args.freq_checkpoint == 0 :
            checkpoint_save(model, save_dir, epoch, validation_loss)
        
        if args.wandb:
            wandb_status = {
                "lr" : now_lr,
                "train_category_acc": train_cat_acc,
                "train_attribute_recall": train_attr_recall,
                "train_attribute": attribute_loss,
                "train_category": category_loss,
                "train_landmark": landmark_loss,
                "train_visibility": visibility_loss,
                "train_loss": train_loss,
                "valid_category_acc": val_cat_acc,
                "valid_attribute_recall": val_attr_recall,
                "valid_attribute": validation_attribute_loss,
                "valid_category": validation_category_loss,
                "valid_landmark": validation_landmark_loss,
                "valid_visibility": validation_visibility_loss,
                "valid_loss": validation_loss,
            }
            
            wandb.log(wandb_status)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/media/jaeho/SSD/datasets/deepfashion/split/')
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--project", type=str, default="TSFashionNet")
    parser.add_argument("--ckpt_savedir", type=str, default="/media/jaeho/HDD/ckpt")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--freq_checkpoint", type=int, default=1)
    
    args = parser.parse_args()
    
    train(args)