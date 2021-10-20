import os

import logging
import pandas as pd 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch import nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR

import torch.nn.functional as F

import json
import gc
from sklearn.metrics import roc_auc_score, accuracy_score

from segmentation_models_pytorch.unetplusplus.model import UnetPlusPlus
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import argparse

parser = argparse.ArgumentParser(description='Insert some arguments')
parser.add_argument('--gpu', type=int,
                    help='GPU ID', default=0)
parser.add_argument('--batch_size', type=int,
                    help='Batch size', default=128)
parser.add_argument('--n_workers', type=int,
                    help='Number of parrallel workers', default=8)
args = parser.parse_args()

with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)

FOLDER = SETTINGS['SEGMENT_DATA_DIR']
META_FILE_PATH = os.path.join(SETTINGS['SEGMENT_DATA_DIR'], 'segment_meta_groupkfold.csv')

RUN_FOLDS = [0]
SEED = 67
DIM = (224, 224, 3)
N_WORKERS = args.n_workers
BATCH_SIZE = args.batch_size
BASE_LR = 1e-4
NUM_EPOCHS = 500
PATIENT = 10
SAMPLE = None
DEVICE = torch.device(f'cuda:{args.gpu}')

PARENT_OUT_FOLDER = f'models/'    

CANDIDATES = [
    {
        'backbone_name':'densenet121',
        'ver_note':'2d_segment',
        'backbone_pretrained':f'pretrained_models/unet_pp_densenet121_2channels_out.pth',
        'batch_size':BATCH_SIZE,
        'warm_up_epochs':10,
    },
]

import sys
from utils.general import seed_torch, init_progress_dict, log_to_progress_dict, save_progress, log_and_print, get_logger

# seed every thing
seed_torch(SEED)

# ================= Some helper functions ====================
class BrainSegment2DDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv, transforms=None):
        self.csv = csv.reset_index(drop=True)
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = np.load(row['file_path']+'.npy')
        image = np.stack([image]*3, axis=-1)
        mask = np.load(row['segfile_path']+'.npy')
        mask = np.stack([mask[0], mask[1]], axis=-1)
        
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.permute(2,0,1)
        
        return image, mask
    

def get_train_transforms(candidate):
    dim = candidate.get('dim', DIM)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
     
            A.Resize(width=dim[1], height=dim[0], always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

def get_valid_transforms(candidate):
    dim = candidate.get('dim', DIM)
    return A.Compose(
        [
            A.Resize(width=dim[1], height=dim[0], always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

def get_model(candidate):
    model = UnetPlusPlus(
        encoder_name = candidate['backbone_name'],
        encoder_depth = 5,
        encoder_weights = None,
        classes = 2,
        activation = 'sigmoid',
    )

    weight_path = candidate.get('backbone_pretrained')
    if(weight_path is not None):
        print('Load pretrained:', weight_path)
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        
    return model


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train_valid_fn(dataloader,model,criterion,iou_metric,optimizer=None,device='cuda:0',
                            scheduler=None,epoch=0, mode='train', scaler=None):
    '''Perform model training'''
    if(mode=='train'):
        model.train()
    elif(mode=='valid'):
        model.eval()
    else:
        raise ValueError('No such mode')
        
    loss_score = AverageMeter()
    iou_score = AverageMeter()
    
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, batch in tk0:
        if(mode=='train'):
            optimizer.zero_grad()
            
        # input, gt
        images, gt_masks = batch
        images = images.to(DEVICE)
        gt_masks = gt_masks.to(DEVICE)

        with torch.cuda.amp.autocast():
            # prediction
            pred_masks = model(images)

            # compute loss
            loss = criterion(y_true=gt_masks, y_pred=pred_masks)
            
            # compute metric
            iou = iou_metric(pred_masks, gt_masks)
        
        if(mode=='train'):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        loss_score.update(loss.detach().cpu().item(), dataloader.batch_size)
        iou_score.update(iou.detach().cpu().item(), dataloader.batch_size)
        
        if(mode=='train'):
            tk0.set_postfix(Loss_Train=loss_score.avg, IOU_Train=iou_score.avg, 
                            Epoch=epoch, LR=optimizer.param_groups[0]['lr'])
        elif(mode=='valid'):
            tk0.set_postfix(Loss_Valid=loss_score.avg, IOU_Valid=iou_score.avg, Epoch=epoch)
        
        del batch, images, gt_masks, pred_masks, loss, iou
        torch.cuda.empty_cache()
        
    if(mode=='train'):
        if(scheduler.__class__.__name__ == 'CosineAnnealingWarmRestarts'):
            scheduler.step(epoch=epoch)
        elif(scheduler.__class__.__name__ == 'ReduceLROnPlateau'):
            scheduler.step(loss_score.avg)
    
    return loss_score.avg, iou_score.avg

def dfs_freeze(module):
    for name, child in module.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
        
def dfs_unfreeze(module):
    for name, child in module.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)
# ===========================================================
        
    
# ================ Read metadata =================
df = pd.read_csv(META_FILE_PATH)
# ================================================


# ============================ Training ==============================
for candidate in CANDIDATES:
    print(f"######################### Candidate: {candidate['backbone_name']} ############################")
    run_folds = candidate.get('run_folds', RUN_FOLDS)
    
    parent_out_folder = candidate.get('parent_out_folder', PARENT_OUT_FOLDER)
    ver_note = candidate['ver_note']
    out_folder_name = f"{candidate['backbone_name']}_{ver_note}"
    out_folder = os.path.join(parent_out_folder, out_folder_name)

    os.makedirs(out_folder, exist_ok=True)
    
    for valid_fold in run_folds:
        # Read data
        if(SAMPLE):
            df = df.sample(SAMPLE, random_state=SEED)

        train_df = df[df.fold!=valid_fold]
        valid_df = df[df.fold==valid_fold]

        print(f'\n\n================= Fold {valid_fold} ==================')
        print(f'Number of training images: {len(train_df)}. Number of valid images: {len(valid_df)}')
        
        
        train_dataset = BrainSegment2DDataset(train_df, get_train_transforms(candidate))
        valid_dataset = BrainSegment2DDataset(train_df, get_valid_transforms(candidate))
        
        batch_size = candidate.get('batch_size', BATCH_SIZE)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=N_WORKERS)
        
        # model
        model = get_model(candidate)
        # freeze layer
        dfs_freeze(model.encoder)
        print(' -------- Start warm up process ----------')
        print('Freeze encoder')
        model.to(DEVICE)
        print()
        
        # Optimizer and scheduler
        base_lr = candidate.get('base_lr', BASE_LR)
        optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)

        num_training_steps = NUM_EPOCHS * len(train_loader)
        lr_scheduler = ReduceLROnPlateau(optimizer=optim)

        # loss
        criterion = DiceLoss(mode='binary', from_logits=False)
        iou_metric = IoU()

        # use amp to accelerate training
        scaler = torch.cuda.amp.GradScaler()

        # Logging
        logger = get_logger(
            name = f'training_log_fold{valid_fold}.txt',
            path=os.path.join(out_folder, f'training_log_fold{valid_fold}.txt')
        )

        best_valid_loss = 9999
        best_valid_ep = 0
        patient = PATIENT

        progress_dict = init_progress_dict(['loss', 'IOU'])

        start_ep = candidate.get('warm_start_ep', 1)
        print('Start ep:', start_ep)

        # warm up epochs
        warm_up_epochs = candidate.get('warm_up_epochs', 1)

        
        for epoch in range(start_ep, NUM_EPOCHS+1):
            if (epoch==warm_up_epochs+1):
                print(' -------- Finish warm up process ----------')
                print('Unfreeze encoder')
                dfs_unfreeze(model.encoder)
                optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)
                lr_scheduler = ReduceLROnPlateau(optimizer=optim)
                
            # =============== Training ==============
            train_loss, train_iou = train_valid_fn(train_loader,model, criterion, iou_metric,
                                                    optimizer=optim,device=DEVICE,
                                                    scheduler=lr_scheduler,epoch=epoch,mode='train',
                                                  scaler=scaler)
            valid_loss, valid_iou = train_valid_fn(valid_loader,model, criterion, iou_metric,
                                                     device=DEVICE, 
                                                     epoch=epoch,mode='valid',
                                                  scaler=scaler)

            current_lr = optim.param_groups[0]['lr']
            log_line = f'Model: {out_folder_name}. Epoch: {epoch}. '
            log_line += f'Train loss:{train_loss} - Valid loss: {valid_loss}. '
            log_line += f'Train IOU:{train_iou} - Valid IOU: {valid_iou}. '
            log_line += f'Lr: {current_lr}.'

            log_and_print(logger, log_line)

            metric_dict = {'train_loss':train_loss,'valid_loss':valid_loss,
                           'train_IOU':train_iou, 'valid_IOU':valid_iou,
                       }

            progress_dict = log_to_progress_dict(progress_dict, metric_dict)

            # plot figure and save the progress chart
            save_progress(progress_dict, out_folder, out_folder_name, valid_fold, show=False)

            if(valid_loss < best_valid_loss):
                best_valid_loss = valid_loss
                best_valid_ep = epoch
                patient = PATIENT # reset patient

                # save model
                name = os.path.join(out_folder, 'Fold%d_%s.pth'%(valid_fold, 
                                                                 out_folder_name, 
                                                                ))
                log_and_print(logger, 'Saving model to: ' + name)
                torch.save(model.state_dict(), name)
            else:
                patient -= 1
                log_and_print(logger, 'Decrease early-stopping patient by 1 due valid loss not decreasing. Patient='+ str(patient))

            if(patient == 0):
                log_and_print(logger, 'Early stopping patient = 0. Early stop')
                break

# ======================================================================


