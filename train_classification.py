import os

import logging
import pandas as pd 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR

import torch.nn.functional as F

from segmentation_models_pytorch.unetplusplus.model import UnetPlusPlus
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import gc
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import argparse

parser = argparse.ArgumentParser(description='Insert some arguments')
parser.add_argument('--mri_type', type=str,
                    help='Train your model on which MRI type. Should be one of: FLAIR, T1w, T1wCE, T2w, All (All means sequentially training the above 4 mri types)', default='FLAIR')
parser.add_argument('--gpu', type=int,
                    help='GPU ID', default=0)
parser.add_argument('--batch_size', type=int,
                    help='Batch size', default=4)
parser.add_argument('--n_workers', type=int,
                    help='Number of parrallel workers', default=8)
args = parser.parse_args()

with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)

DATA_FOLDER = SETTINGS['CLASSIFICATION_DATA_DIR']
META_FILE_PATH = f'{DATA_FOLDER}/meta_classification.csv'
KFOLD_FILE_PATH = SETTINGS['KFOLD_PATH']

RUN_FOLDS = [0]
MRI_TYPES = ['FLAIR','T1w', 'T1wCE', 'T2w'] if args.mri_type == 'All' else [args.mri_type]
STRIDE = 5
SEQ_LEN = 35
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 1
SEED = 67
DIM = (224, 224, 3)
N_WORKERS = args.n_workers
BATCH_SIZE = args.batch_size
BASE_LR = 1e-6
NUM_EPOCHS = 80
PATIENT = 10
SAMPLE = None
DEVICE = torch.device(f'cuda:{args.gpu}')

PARENT_OUT_FOLDER = 'models/'   

CANDIDATES = [
    {
        'backbone_name':'eca_nfnet_l0',
        'ver_note':'2d_classification',
        'backbone_pretrained':'pretrained_models/eca_nfnet_l0.pth',
        'batch_size':BATCH_SIZE,
        'warm_up_epochs':5,
    },
]


import sys
from utils.general import seed_torch, init_progress_dict, log_to_progress_dict, save_progress, log_and_print, get_logger

# seed every thing
seed_torch(SEED)


def chunk_slices(list_files):
    list_files = sorted(list_files)
    chunks = []
    n_chunks = max(int(np.ceil((len(list_files) - SEQ_LEN) / STRIDE ) + 1),1)
    for i in range(n_chunks):
        s = i*STRIDE
        e = min(s+SEQ_LEN, len(list_files))
        chunks.append(list_files[s:e])
    return chunks

def expand(row):
    list_files = row['chunk_file_paths']
    return pd.DataFrame({
        'BraTS21ID':[row['BraTS21ID']]*len(list_files),
        'MGMT_value':[row['MGMT_value']]*len(list_files),
        'mri_type':[row['mri_type']]*len(list_files),
        'file_path':list_files,
        'fold':[row['fold']]*len(list_files)
    })

def get_first_value(df, col_name):
    df[col_name] = df[col_name].map(lambda x: list(x)[0])

    
def process_df_mri_type(df_mri):
    df_mri_group = df_mri.groupby('BraTS21ID').agg(list)
    df_mri_group = df_mri_group.reset_index()
    df_mri_group['chunk_file_paths'] = df_mri_group.file_path.map(chunk_slices)
    df_mri_group['chunk_count'] = df_mri_group['chunk_file_paths'].map(lambda x: len(x))
    df_mri_group['chunk_cum_count'] = df_mri_group['chunk_count'].cumsum()
    df_mri_group_expand = df_mri_group.apply(expand, axis=1).tolist()
    df_mri_group_expand = pd.concat(df_mri_group_expand)

    for col_name in ['MGMT_value', 'mri_type', 'fold']:
        get_first_value(df_mri_group_expand, col_name)
        
    return df_mri_group_expand    
    
class BrainClassification2DDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv, transforms=None):
        self.csv = csv.reset_index(drop=True)
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        list_file_path = row['file_path']
        list_images = []
        label = row['MGMT_value']
        for i, path in enumerate(list_file_path):
            image = np.load(path)
            label = row['MGMT_value']
            list_images.append(image)
                
        images = np.stack(list_images, axis=0)
        if(images.shape[0] < SEQ_LEN):
            n_pad = SEQ_LEN - images.shape[0]
            pad_matrix = np.zeros(shape=(n_pad, images.shape[1], images.shape[2], images.shape[3]))
            images = np.concatenate([images, pad_matrix], axis=0)
            
        if self.augmentations:
            images_dict = dict()
            for i in range(len(images)):
                if(i==0):
                    images_dict['image'] = images[i]
                else:
                    images_dict[f'image{i-1}'] = images[i]
            augmented = self.augmentations(**images_dict)
            
            transformed_images = []
            for i in range(len(images)):
                if(i==0):
                    transformed_images.append(augmented['image'])
                else:
                    transformed_images.append(augmented[f'image{i-1}'])
                    
            transformed_images = np.stack(transformed_images, axis=0)
            return transformed_images, torch.tensor(label)
            
        return images, torch.tensor(label)
    
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms(candidate):
    dim = candidate.get('dim', DIM)
    seq_len = candidate.get('seq_len', SEQ_LEN)
    additional_targets = {f'image{i}':'image' for i in range(SEQ_LEN-1)}
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            
            A.Resize(width=dim[1], height=dim[0], always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ],
        additional_targets=additional_targets
    )

def get_valid_transforms(candidate):
    dim = candidate.get('dim', DIM)
    additional_targets = {f'image{i}':'image' for i in range(SEQ_LEN-1)}
    return A.Compose(
        [
            A.Resize(width=dim[1], height=dim[0], always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ],
        additional_targets=additional_targets
    )    

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

import timm

class BrainSequenceModelNFNet(nn.Module):
    def __init__(self, backbone_name, backbone_pretrained,
                 lstm_dim=64, lstm_layers=1, lstm_dropout=0., 
                 n_classes=1):
        super(BrainSequenceModelNFNet, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        self.backbone.load_state_dict(torch.load(backbone_pretrained))
        
        lstm_inp_dim = self.backbone.head.fc.in_features
        
        self.backbone.head.fc = nn.Identity()
        
        self.lstm = nn.LSTM(lstm_inp_dim, lstm_dim, num_layers=lstm_layers, 
                            batch_first=True, bidirectional=True,
                            dropout=lstm_dropout)
        
        self.clf_head = nn.Linear(lstm_dim*2*SEQ_LEN, n_classes)
        
    def forward(self, x):
        n = x.shape[0]
        seq_length = x.shape[1]
        concat_x = torch.cat([x[i] for i in range(n)], axis=0)
        concat_x = self.backbone(concat_x)
        
        
        stacked_x = torch.stack([concat_x[i*seq_length:i*seq_length+seq_length] for i in range(n)], axis=0)
        
        seq_features, _ = self.lstm(stacked_x)
        seq_features = seq_features.reshape(n,-1)
        
        logits = self.clf_head(seq_features)
        
        return logits


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
        
def train_valid_fn(dataloader,model, criterion, scaler, optimizer=None,device='cuda:0',scheduler=None,
                   epoch=0,mode='train', metric='auc'):
    '''Perform model training'''
    if(mode=='train'):
        model.train()
    elif(mode=='valid'):
        model.eval()
    else:
        raise ValueError('No such mode')
        
    loss_score = AverageMeter()
    
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    all_predictions = []
    all_labels = []
    for i, batch in tk0:
        if(mode=='train'):
            optimizer.zero_grad()
            
        # input, gt
        voxels, labels = batch
        voxels = voxels.to(device)
        labels = labels.to(device).float()

        # prediction
        with torch.cuda.amp.autocast():
            logits = model(voxels)
            logits = logits.view(-1)
            probs = logits.sigmoid()
            # compute loss
            loss = criterion(logits, labels)
        
        if(mode=='train'):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        loss_score.update(loss.detach().cpu().item(), dataloader.batch_size)

        # append for metric calculation
        all_predictions.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        
        if(mode=='train'):
            tk0.set_postfix(Loss_Train=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])
        elif(mode=='valid'):
            tk0.set_postfix(Loss_Valid=loss_score.avg, Epoch=epoch)
        
        del batch, voxels, labels, logits, probs, loss
        torch.cuda.empty_cache()

    if(mode=='train'):
        if(scheduler.__class__.__name__ == 'CosineAnnealingWarmRestarts'):
            scheduler.step(epoch=epoch)
        elif(scheduler.__class__.__name__ == 'ReduceLROnPlateau'):
            scheduler.step(loss_score.avg)

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    if(metric == 'auc'):
        auc = roc_auc_score(y_true=all_labels, y_score=all_predictions)
        return loss_score.avg, auc 
    
    return loss_score.avg

    
# ============ Read metadata ==============    
df = pd.read_csv(META_FILE_PATH)
kfold_df = pd.read_csv(KFOLD_FILE_PATH)
df = df.merge(kfold_df, on='BraTS21ID')

df_flair = df[df.mri_type=='FLAIR']
df_t1 = df[df.mri_type=='T1w']
df_t1ce = df[df.mri_type=='T1wCE']
df_t2 = df[df.mri_type=='T2w']
# =========================================


# ================================ Training ==================================
for candidate in CANDIDATES:
    print(f"######################### Candidate: {candidate['backbone_name']} ############################")
    run_folds = candidate.get('run_folds', RUN_FOLDS)
    
    parent_out_folder = candidate.get('parent_out_folder', PARENT_OUT_FOLDER)
    ver_note = candidate['ver_note']

    for mri_type in MRI_TYPES:
        out_folder_name = f"{candidate['backbone_name']}_{ver_note}"
        out_folder = os.path.join(parent_out_folder, out_folder_name, mri_type)
        os.makedirs(out_folder, exist_ok=True)
    
        for valid_fold in run_folds:
            # Read data
            if(SAMPLE):
                df = df.sample(SAMPLE, random_state=SEED)
            if(mri_type != 'all'):
                df_mri = df[df.mri_type==mri_type]
            
            # process data
            df_mri = process_df_mri_type(df_mri)
                
            train_df = df_mri[df_mri.fold!=valid_fold]
            valid_df = df_mri[df_mri.fold==valid_fold]

            print(f'\n\n================= Fold {valid_fold}. MRI: {mri_type} ==================')
            print(f'Number of training samples: {len(train_df)}. Number of valid samples: {len(valid_df)}')

            # train and valid transforms
            train_transforms = get_train_transforms(candidate)
            valid_transforms = get_valid_transforms(candidate)

            # create data loader
            train_dataset =  BrainClassification2DDataset(train_df, train_transforms)
            valid_dataset = BrainClassification2DDataset(valid_df, valid_transforms)

            batch_size = candidate.get('batch_size', BATCH_SIZE)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=N_WORKERS, pin_memory=torch.cuda.is_available())
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=N_WORKERS, pin_memory=torch.cuda.is_available())


            # Model
            model = BrainSequenceModelNFNet(candidate['backbone_name'], 
                                            candidate['backbone_pretrained'],
                                           lstm_dim=LSTM_HIDDEN_SIZE,lstm_layers=LSTM_LAYERS)
            model.to(DEVICE)
            print()

            warm_start_weight = candidate.get('warm_start_weight')
            if(warm_start_weight):
                print('Load warm start weight:', warm_start_weight)

            # freeze pretrained layers
            dfs_freeze(model.backbone)
            print(' -------- Start warm up process ----------')
            print('Freeze backbone')
            model = model.to(DEVICE)
            print()


            # Optimizer and scheduler
            base_lr = candidate.get('base_lr', BASE_LR)
            optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)

            num_training_steps = NUM_EPOCHS * len(train_loader)
            lr_scheduler = ReduceLROnPlateau(optimizer=optim, factor=0.67, patience=3, verbose=True)

            # loss
            criterion = nn.BCEWithLogitsLoss()


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

            progress_dict = init_progress_dict(['loss', 'AUC'])

            start_ep = candidate.get('warm_start_ep', 1)
            print('Start ep:', start_ep)

            # warm up epochs
            warm_up_epochs = candidate.get('warm_up_epochs', 0)


            for epoch in range(start_ep, NUM_EPOCHS+1):
                if(epoch==warm_up_epochs+1):
                    print(' -------- Finish warm up process ----------')
                    print('Unfreeze backbone')
                    dfs_unfreeze(model.backbone)
                    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)
                    lr_scheduler = ReduceLROnPlateau(optimizer=optim)

                # =============== Training ==============
                train_loss, train_auc = train_valid_fn(train_loader,model,criterion, scaler, optimizer=optim,device=DEVICE,
                                            scheduler=lr_scheduler,epoch=epoch,mode='train', metric='auc')
                valid_loss, valid_auc = train_valid_fn(valid_loader,model,criterion, scaler, device=DEVICE,epoch=epoch,mode='valid', metric='auc')

                current_lr = optim.param_groups[0]['lr']
                log_line = f'Model: {out_folder_name}. Epoch: {epoch}. '
                log_line += f'Train loss:{train_loss} - Valid loss: {valid_loss}. '
                log_line += f'Train AUC:{train_auc} - Valid AUC: {valid_auc}. '
                log_line += f'Lr: {current_lr}.'

                log_and_print(logger, log_line)

                metric_dict = {'train_loss':train_loss,'valid_loss':valid_loss,
                               'train_AUC':train_auc, 'valid_AUC':valid_auc,
                           }

                progress_dict = log_to_progress_dict(progress_dict, metric_dict)

                # plot figure and save the progress chart
                save_progress(progress_dict, out_folder, out_folder_name, valid_fold, show=False)

                if(valid_loss < best_valid_loss):
                    best_valid_loss = valid_loss
                    best_valid_ep = epoch
                    patient = PATIENT # reset patient

                    # save model
                    name = os.path.join(out_folder, f'%s_Fold%d_%s.pth'%(mri_type, valid_fold, out_folder_name))
                    log_and_print(logger, 'Saving model to: ' + name)
                    torch.save(model.state_dict(), name)
                else:
                    patient -= 1
                    log_and_print(logger, 'Decrease early-stopping patient by 1 due valid loss not decreasing. Patient='+ str(patient))

                if(patient == 0):
                    log_and_print(logger, 'Early stopping patient = 0. Early stop')
                    break
# =============================================================================