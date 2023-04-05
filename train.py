import os
import time
import wandb
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import config, upper_class_name, sweep_configuration
from dataset import TSDataset 
from model import TSFashionNet
from bit_model import BiT_TSFashionNet, PreTrained_Dict
from square_pad import SquarePad

from custom_loss import LandmarkLoss  # 변경
from base_loss import BaseLoss
from cov_weighting_loss import CoVLoss

from utils import fix_seed, get_now, checkpoint_save, make_metric_dict
from utils import calc_class_recall, calc_metric, print_config

fix_seed()

def train():
    
    batch_size = config['batch_size']
    lr = config['lr']
    epochs = config['epochs']
    shape_lr = config['shape_lr']
    shape_epochs = config['shape_epochs']
    img_dir = config['img_dir']
    cov_epoch = config['cov_epoch']
    att_weight = config['att_weight']    
    loss_mode = config['loss_mode']
    
    # wandbv init ####################
    if config['name']:
        wandb_name = f"{config['name']}_{get_now(time=True)}"
    else :
        wandb_name = f"{config['backbone']}_TSFashionNet_{get_now(time=True)}"
            
    if config['wandb']:
        if config['sweep']:
            config['scheduler'] = 'plat'
            wandb.init(entity='ljh415', project=config['project'], dir='/media/jaeho/HDD/wandb/',
                       config=config)
            w_config = wandb.config
            batch_size = w_config.batch_size
            lr = w_config.lr
            epochs = w_config.epochs
            shape_lr = w_config.shape_lr
            shape_epochs = w_config.shape_epochs
            
        else:
            wandb.init(entity="ljh415", project=config['project'], dir='/media/jaeho/HDD/wandb/',
                   name=wandb_name, config=config)
    
    # checkpoint save directory
    save_dir = os.path.join(config['ckpt_savedir'], config['project'], wandb_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    train_path = os.path.join(config['data_path'], 'train.pickle')
    valid_path = os.path.join(config['data_path'], 'valid.pickle')
    num_workers = config['num_workers']
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
    train_dataset = TSDataset(train_path, img_dir, transform=train_transform)
    valid_dataset = TSDataset(valid_path, img_dir, transform=val_transform)
    
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
    
    if loss_mode == 'base':
        criterion_dict = {
            'base': BaseLoss(config['reduction'])
        }
    elif loss_mode == 'cov':
        criterion_dict = {
            'shape': CoVLoss(config['reduction'], shape_only=True),
            'all' : CoVLoss(config['reduction'], shape_only=False)
        }
    elif loss_mode == 'multi':
        criterion_dict = {
            'cov': CoVLoss(config['reduction'], shape_only=False),
            'base': BaseLoss(config['reduction'])
        }
    else :
        raise Exception("Wrong loss mode.")
    
    # model
    if config['backbone'] == 'vgg':
        model = TSFashionNet().to(device)
        shape_optimizer = torch.optim.Adam(model.parameters(), lr=shape_lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config['backbone'] == 'bit' :
        model = BiT_TSFashionNet(model_name=config['bit_model_name']).to(device)
        shape_optimizer = torch.optim.SGD(model.parameters(), lr=shape_lr, momentum=0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # scheduler
    if config['sweep'] or config['scheduler']=='plat':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.decay_factor,
                                                                  patience=args.patience, verbose=True)
    else :
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config['milestones'], gamma=0.1, verbose=True)
    
    if config['sweep']:
        print_config(w_config, model, trainable=True)
    else :
        print_config(config, model, trainable=True)
    
    print(f"{'='*20} training only shape {'='*20}")
    
    #### shape먼저 3epoch
    for epoch in range(shape_epochs):
        model.train()
        
        step = 0
        running_shape_loss = 0
        running_landmark_loss, running_visibility_loss = 0, 0
        
        for batch_idx, (img_batch, _, _, visibility_batch, landmark_batch) in enumerate(train_dataloader):
            
            shape_optimizer.zero_grad()
            
            img_batch = img_batch.to(device)
            landmark_batch = landmark_batch.to(device)
            visibility_batch = visibility_batch.to(device)
            
            vis_out, loc_out = model(img_batch, shape=True)  # training only shape biased stream
            
            if loss_mode == 'cov':
                loss, vis_loss, lm_loss, _ = criterion_dict['shape'](
                    preds=(vis_out, loc_out),
                    targets=(visibility_batch, landmark_batch),
                    mode='train'
                )
            else : # loss_mode == 'base'
                vis_loss, lm_loss = criterion_dict['base'](
                    preds = (vis_out, loc_out),
                    targets = (visibility_batch, landmark_batch),
                    shape = True
                )
                
                loss = vis_loss + lm_loss
            
            # loss calc
            running_shape_loss += loss.detach().cpu().numpy().item()
            running_landmark_loss += lm_loss.detach().cpu().numpy().item()
            running_visibility_loss += vis_loss.detach().cpu().numpy().item()
            
            loss.backward()
            shape_optimizer.step()
            step += 1
            
            status = (
                "\r> epoch: {:3d} > step: {:3d} > loss: {:.3f}, lr: {} ".format(
                                epoch+1,
                                step,
                                running_shape_loss/(batch_idx+1),
                                shape_lr,
                            )
            )
            
            if args.logging_shape_train:
                if args.wandb and batch_idx % 5 == 0 :
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
                
                if loss_mode == 'cov':
                    val_loss, vis_val_loss, lm_val_loss = criterion_dict['shape'](
                        preds=(vis_out, loc_out),
                        targets=(visibility_batch, landmark_batch),
                        mode='valid'
                    )
                else :  # loss_mode == 'base'
                    vis_val_loss, lm_val_loss = criterion_dict['base'](
                        preds = (vis_out, loc_out),
                        targets = (visibility_batch, landmark_batch),
                        shape = True
                    )
                    
                    val_loss = vis_val_loss + lm_val_loss
                
                running_val_loss += val_loss.item()
                running_landmark_val_loss += lm_val_loss.item()
                running_visibility_val_loss += vis_val_loss.item()

                if args.logging_shape_train:
                    if args.wandb and batch_idx % 5 == 0 :
                        wandb.log({
                            "shape_stream-val_lm_loss": running_landmark_val_loss/(batch_idx+1),
                            "shape_stream-val_vis_loss": running_visibility_val_loss/(batch_idx+1),
                            "shape_stream-val_total_loss": running_val_loss/(batch_idx+1),
                        })
                
        validation_loss = running_val_loss / len(valid_dataloader)
        
        print("Validation loss : {:3f}\n".format(validation_loss))
        
        if args.logging_shape_train:
            # if epoch % args.freq_checkpoint == 0:
            #     checkpoint_save(model, save_dir, epoch, validation_loss)
            if args.wandb:
                wandb.log({
                    "train_loss": running_shape_loss / len(train_dataloader),
                    "train_vis_loss": running_visibility_loss / len(train_dataloader),
                    "train_lm_loss": running_landmark_loss / len(train_dataloader),
                    "val_loss": validation_loss,
                    "val_vis_loss": running_visibility_val_loss / len(valid_dataloader),
                    "val_lm_loss": running_landmark_val_loss / len(valid_dataloader),
                })
        
    ###### first 3 epochs, train only shape biased stream

    print(f"{'='*20} training all model {'='*20}")
    
    ## train all stream
    for epoch in range(epochs):
        model.train()
        
        step = 0
        
        running_train_category_acc3, running_train_category_acc5 = 0, 0
        running_train_attr_recall3, running_train_attr_recall5 = 0, 0
        
        if loss_mode != 'base':
            running_loss_weights = torch.zeros(4)
        
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
            
            category_out, attr_out, vis_out, loc_out = model(img_batch, shape=False)
            
            if loss_mode == 'cov':
                loss, cat_loss, att_loss, vis_loss, lm_loss, loss_weights = criterion_dict['all'](
                    preds = (category_out, attr_out, vis_out, loc_out),
                    targets = (category_batch, attribute_batch, visibility_batch, landmark_batch),
                    mode='train'
                )
                running_loss_weights += loss_weights.detach().cpu()
                
            elif loss_mode == 'multi':
                if epoch < cov_epoch:
                    loss, cat_loss, att_loss, vis_loss, lm_loss, loss_weights = criterion_dict['cov'](
                        preds = (category_out, attr_out, vis_out, loc_out),
                        targets = (category_batch, attribute_batch, visibility_batch, landmark_batch),
                        mode='train'
                    )
                    # sum loss-weight
                    running_loss_weights += loss_weights.detach().cpu()
                    
                else :
                    cat_loss, att_loss, vis_loss, lm_loss = criterion_dict['base'](
                        preds = (category_out, attr_out, vis_out, loc_out),
                        targets = (category_batch, attribute_batch, visibility_batch, landmark_batch),
                    )
                    # att_weight=500
                    loss = cat_loss + att_weight*att_loss + lm_loss + vis_loss
                pass
            
            else : # loss_mode == 'base'
                cat_loss, att_loss, vis_loss, lm_loss = criterion_dict['base'](
                        preds = (category_out, attr_out, vis_out, loc_out),
                        targets = (category_batch, attribute_batch, visibility_batch, landmark_batch),
                    )
                loss = cat_loss + 500*att_loss + lm_loss + vis_loss
            
            # loss calc
            running_train_loss += loss.detach().cpu().numpy().item()
            running_landmark_loss += lm_loss.detach().cpu().numpy().item()
            running_visibility_loss += vis_loss.detach().cpu().numpy().item()
            running_category_loss += cat_loss.detach().cpu().numpy().item()
            running_attribute_loss += att_loss.detach().cpu().numpy().item()
            
            loss.backward()
            optimizer.step()
            step += 1
            
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
        
        if (epoch < cov_epoch) and (loss_mode != 'base'):
            train_loss_weights = running_loss_weights / len(train_dataloader)
        
        train_cat_acc3 = running_train_category_acc3 / len(train_dataloader)
        train_cat_acc5 = running_train_category_acc5 / len(train_dataloader)
        
        train_attr_recall3 = running_train_attr_recall3 / len(train_dataloader)
        train_attr_recall5 = running_train_attr_recall5 / len(train_dataloader)
        
        ## validate  #########
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
                
                category_out, attr_out, vis_out, loc_out = model(img_batch, shape=False)
                
                if loss_mode == 'cov':
                    val_loss, cat_val_loss, att_val_loss, vis_val_loss, lm_val_loss = criterion_dict['all'](
                        preds = (category_out, attr_out, vis_out, loc_out),
                        targets = (category_batch, attribute_batch, visibility_batch, landmark_batch),
                        mode='valid'
                    )
                    
                elif loss_mode == 'multi':
                    if epoch < cov_epoch:
                        val_loss, cat_val_loss, att_val_loss, vis_val_loss, lm_val_loss = criterion_dict['cov'](
                            preds = (category_out, attr_out, vis_out, loc_out),
                            targets = (category_batch, attribute_batch, visibility_batch, landmark_batch),
                            mode='valid'
                        )
                    else:
                        cat_val_loss, att_val_loss, vis_val_loss, lm_val_loss = criterion_dict['base'](
                            preds = (category_out, attr_out, vis_out, loc_out),
                            targets = (category_batch, attribute_batch, visibility_batch, landmark_batch),
                        )
                    
                        val_loss = cat_val_loss + 500*att_val_loss + lm_val_loss + vis_val_loss
                    
                else :  # loss_mode == 'base'
                    cat_val_loss, att_val_loss, vis_val_loss, lm_val_loss = criterion_dict['base'](
                            preds = (category_out, attr_out, vis_out, loc_out),
                            targets = (category_batch, attribute_batch, visibility_batch, landmark_batch),
                        )
                    
                    val_loss = cat_val_loss + 500*att_val_loss + lm_val_loss + vis_val_loss
                
                running_val_loss += val_loss.item()
                running_landmark_val_loss += lm_val_loss.item()
                running_visibility_val_loss += vis_val_loss.item()
                running_category_val_loss += cat_val_loss.item()
                running_attribute_val_loss += att_val_loss.item()
                
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
        
        if epoch >= config['cov_epoch'] :
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
            if loss_mode == "cov":
                wandb_status['cat_loss_weight'] = train_loss_weights[0]
                wandb_status['att_loss_weight'] = train_loss_weights[1]
                wandb_status['vis_loss_weight'] = train_loss_weights[2]
                wandb_status['lm_loss_weight'] = train_loss_weights[3]
            if (epoch < cov_epoch) and (loss_mode != 'base'):
                wandb_status['cat_loss_weight'] = train_loss_weights[0]
                wandb_status['att_loss_weight'] = train_loss_weights[1]
                wandb_status['vis_loss_weight'] = train_loss_weights[2]
                wandb_status['lm_loss_weight'] = train_loss_weights[3]
            
            wandb.log(wandb_status)

def test():
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if config['backbone'] == 'vgg':
        model = TSFashionNet().to(device)
    elif config['backbone'] == 'bit':
        model = BiT_TSFashionNet(model_name=config['bit_model_name']).to(device)
    
    ckpt_dict = torch.load(config['ckpt'])
    model.load_state_dict(ckpt_dict['model_state_dict'])
    model.eval()
    
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = TSDataset(
        data_path=os.path.join(config['data_path'], 'test.pickle'),
        img_dir = config['img_dir'],
        transform=trans
    )
    test_dataloder = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    if config['test_num'] is None:
        config['test_num'] = len(test_dataset)
        
    # set metric
    metric_dict = make_metric_dict()
    result_dict = defaultdict(lambda : defaultdict(list))
    class_recall_dict = defaultdict(lambda : defaultdict(int))
    
    # inference, calc
    total_time = 0
    for idx, (img, cat, att, _, _) in enumerate(test_dataloder):
        one_iter_start = time.time()
        
        cat = torch.Tensor([torch.squeeze(cat)])
        cat = cat.to(device)
        att = att.to(device)
        img_tensor = img.to(device)
        
        cat_out, att_out, _, _ = model(img_tensor, shape=False)
        
        # calc metric
        calc_dict = calc_metric(metric_dict, cat_out, att_out, cat, att)
        
        for key, score in calc_dict.items():
            task, metric_name = key.split("-")
            result_dict[task][metric_name].append(score)
        
        # calc recall about classes
        tp_dict = calc_class_recall(att, att_out)
        for key, values in tp_dict.items():
            for value in values:
                class_recall_dict[key][value] += 1
        
        one_iter_time = time.time() - one_iter_start
        total_time += one_iter_time
        avg_time = total_time / (idx+1)
        estimate_time = avg_time * config['test_num']
        
        total_show = str(datetime.timedelta(seconds=total_time))
        estimate_show = str(datetime.timedelta(seconds=estimate_time))
        total_short = total_show.split(".")[0]
        estimate_short = estimate_show.split(".")[0]
        
        status = (
            "\r {:6d}/{:6d}\t[{} => {}] | top3_acc: {:.3f}, top5_acc: {:.3f}, top3_recall: {:.3f}, top5_recall: {:.3f}  ".format(
                idx+1,
                config['test_num'],
                total_short,
                estimate_short,
                np.mean(result_dict['category']['top3_acc']),
                np.mean(result_dict['category']['top5_acc']),
                np.mean(result_dict['attribute']['top3_recall']),
                np.mean(result_dict['attribute']['top5_recall']),
            )
        )
        print(status, end="")
        
        if idx == config['test_num']:
            break
    print()
    # show
    print_config(config, model, trainable=True)
    
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
    parser.add_argument("--img_dir", type=str, default='/media/jaeho/SSD/datasets/deepfashion/img-001/')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--shape_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--shape_lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--project", type=str, default="TSFashionNet")
    parser.add_argument("--ckpt_savedir", type=str, default="/media/jaeho/HDD/ckpt")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--freq_checkpoint", type=int, default=1)
    parser.add_argument("--logging_shape_train", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--test_num", type=int, default=None)
    parser.add_argument("--backbone", type=str, default='vgg')
    parser.add_argument("--milestones", type=str, default='6')
    parser.add_argument("--reduction", type=str, default='sum')
    parser.add_argument("--bit_model_name", type=str, default=None)
    parser.add_argument("--sweep", action='store_true')
    parser.add_argument("--sweep_count", type=int, default=3)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--decay_factor", type=float, default=0.1)
    parser.add_argument("--cov_epoch", type=int, default=2)
    parser.add_argument("--att_weight", type=float, default=500.0)
    
    parser.add_argument("--loss_mode", type=str, default='base', help="base, multi, cov")
    
    args = parser.parse_args()
    args.milestones = list(map(int, args.milestones.split(',')))
    
    if args.loss_mode == 'multi' and args.shape_epochs != 0:
        raise Exception("Invalid arguments, 'multi_loss' mode, shape_epochs != 0")
    
    if args.backbone == 'bit':
        if args.bit_model_name is None:
            raise "Check bit_model_name argument"
        else :
            args.bit_model_name = PreTrained_Dict[args.bit_model_name]
    else :
        args.bit_model_name = None
        
    if args.sweep:
        args.wandb = True
    
    config.update(vars(args))
    
    if args.mode == 'train':
        if args.sweep:
            sweep_id = wandb.sweep(sweep=sweep_configuration[args.backbone], project=args.project)
            wandb.agent(sweep_id=sweep_id, function=train, count=args.sweep_count)
        else:
            train()
    elif args.mode == 'test':
        if args.ckpt is None:
            raise Exception("Check the checkpoint path")
        test()
