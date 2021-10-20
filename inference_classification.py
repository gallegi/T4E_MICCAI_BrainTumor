import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

import os
from tqdm.notebook import tqdm

import pydicom
import glob
import sys

from segmentation_models_pytorch.unetplusplus.model import UnetPlusPlus
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU

import time
from multiprocessing import Pool

import json
import argparse

parser = argparse.ArgumentParser(description='Insert some arguments')
parser.add_argument('--gpu', type=int,
                    help='GPU ID', default=0)
parser.add_argument('--classification_batch_size', type=int,
                    help='Classification batch size', default=4)
parser.add_argument('--segment_batch_size', type=int,
                    help='Segmentation batch size', default=64)
parser.add_argument('--n_workers', type=int,
                    help='Number of parrallel workers', default=8)
args = parser.parse_args()

with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)

DICOM_IM_FOLDER = os.path.join(SETTINGS['DICOM_DATA_DIR'], 'test')
IM_FOLDER = SETTINGS['TEMP_DATA_DIR']
DEVICE = f'cuda:{args.gpu}'
N_WORKERS = 4
STRIDE = 5
SEQ_LEN = 35
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 1
MRI_TYPES = ['T1w', 'T1wCE', 'T2w', 'FLAIR']

DIM = (224,224,3)

SEG_BATCH_SIZE = args.segment_batch_size
CLF_BATCH_SIZE = args.classification_batch_size

FAST_SUB = True

SEG_MODEL = {
        'backbone_name':'densenet121',
        'model_path':os.path.join(SETTINGS['SEGMENT_MODEL_DIR'], 'Fold0_densenet121_2d_segment.pth')
    }

CLF_CANDIDATES = [
    {
        'backbone_name':'eca_nfnet_l0',
        'model_path':os.path.join(SETTINGS['CLASSIFICATION_MODEL_DIR'], 'FLAIR' ,'FLAIR_Fold0_eca_nfnet_l0_2d_classification.pth'),
        'mri_type':"flair"
    },
#     {
#         'backbone_name':'eca_nfnet_l0',
#         'model_path':'../input/brain2dclflstm/Fold0_eca_nfnet_l0_2d_clf_v6_lstm_data_v5_ValidLoss0.662_ValidAUC0.642_Ep09.pth',
#         'mri_type':"t1"
#     },
#    {
#         'backbone_name':'eca_nfnet_l0',
#         'model_path':'../input/brain2dclflstm/Fold0_eca_nfnet_l0_2d_clf_v6_lstm_data_v5_ValidLoss0.684_ValidAUC0.555_Ep09.pth',
#         'mri_type':"t1ce"
#     },
#     {
#         'backbone_name':'eca_nfnet_l0',
#         'model_path':'../input/brain2dclflstm/Fold0_eca_nfnet_l0_2d_clf_v6_lstm_data_v5_ValidLoss0.604_ValidAUC0.523_Ep08.pth',
#         'mri_type':"t2"
#     },
]




def get_seg_model(candidate):
    model = UnetPlusPlus(
        encoder_name = candidate['backbone_name'],
        encoder_depth = 5,
        encoder_weights = None,
        classes = 2,
        activation = 'sigmoid',
    )

    weight_path = candidate.get('pretrained_weight')
    if(weight_path is not None):
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        
    return model

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_transform(candidate, spatial_only=False):
    dim = candidate.get('dim', DIM)
    list_trans = [
                A.Resize(width=int(dim[1]*1.2), height=int(dim[0]*1.2), always_apply=True),
                A.CenterCrop(width=dim[1], height=dim[0], always_apply=True),
                A.Normalize(), 
                ToTensorV2(p=1.0)
    ]
    return A.Compose(list_trans)

def get_inv_transform(original_w, original_h, candidate):
    dim = candidate.get('dim', DIM)
    list_trans = [
                A.PadIfNeeded(min_height=int(dim[1]*1.2), min_width=int(dim[1]*1.2), always_apply=True),
                A.Resize(width=original_w, height=original_h, always_apply=True),
    ]
    return A.Compose(list_trans)

def normalize_voxels(voxels):
    _min = voxels.min()
    _max = voxels.max()
    new_voxels = (voxels - _min) / (_max-_min) * 255.0
    return new_voxels

def check_empty(img, min_avg=0.1):
    _mean = np.where(img>0, 1, 0).mean()
    if(_mean > min_avg):
        return True
    return False
def find_largest_countours(contours):
    max_cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))
    return max_cnt

def has_good_features(image, mask, area_mask_over_image_min_ratio=0.1, max_count_mask_contours=5):
    _, image_thresh = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
    image_contours, _ = cv2.findContours(image=image_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    max_image_cnt = find_largest_countours(image_contours)
    
    _, mask_thresh = cv2.threshold(mask,0.5,1,cv2.THRESH_BINARY)
    mask_contours, _ = cv2.findContours(image=mask_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    count_n_mask_contours = len(mask_contours)
    if(count_n_mask_contours == 0):
        return False
    max_mask_cnt = find_largest_countours(mask_contours)
    
    area_mask_over_image_ratio = cv2.contourArea(max_mask_cnt) / cv2.contourArea(max_image_cnt)
    
    if(area_mask_over_image_ratio > area_mask_over_image_min_ratio \
       and count_n_mask_contours <= max_count_mask_contours):
        return True
    else:
        return False

def batch_predict_mask(data_loader, model):
    batch_out = []
    for batch_input in data_loader:
        batch_input = batch_input.to(DEVICE)
        batch_out.append(model(batch_input).cpu().detach().numpy())
        
    batch_out = np.concatenate(batch_out, axis=0)
    batch_out = (batch_out > 0.5).astype('uint8')
    
    del batch_input
    torch.cuda.empty_cache()
    
    return batch_out

class BrainSegmentationInferDataset(torch.utils.data.Dataset):
    
    def __init__(self, all_mri_voxels, transforms):
        self.all_mri_voxels = all_mri_voxels
        self.augmentations = transforms

    def __len__(self):
        return len(self.all_mri_voxels)

    def __getitem__(self, index):
        image = self.all_mri_voxels[index]
        image = np.stack([image]*3, axis=-1)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        return image

def error(e):
    print(e)
    
def read_and_preprocess_voxels_update(args):
    if(args!=[]):
        voxels, mri_type, images = args
        global all_transformed_images, corresponding_mri_types, all_images
        all_transformed_images += [image for image in voxels]
        corresponding_mri_types += [mri_type]*len(voxels)
        all_images += images

def read_and_preprocess_voxels(patient_id, mri_type, ext='.dcm'):
    paths = glob.glob(os.path.join(DICOM_IM_FOLDER, patient_id, mri_type, '*'+ext))
    paths = sorted(paths, key=lambda x: int(x.replace(ext,'').split("-")[-1]))
    positions = []
    images = []

    for path in paths:
#         print(path)
        img = pydicom.dcmread(str(path))
        img = img.pixel_array
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if(check_empty(img)):
            images.append(img)

    if(len(images) == 0):
        print("Found no images in case (patient_id, mri, path):", patient_id, mri_type, paths)
        return []

    voxels = np.array(images)
    voxels = normalize_voxels(voxels)  # normalize voxels to range(0,255)
#     print(len(voxels))
    return voxels, mri_type, list(voxels.astype('uint8'))
        
    
def sampling_one_image(patient_id, slice_index, image, out, mri_type):

    mask_0, mask_1 = out[0], out[1]
    inv_transforms = get_inv_transform(image.shape[1], image.shape[0], SEG_MODEL)
    mask_0_original_size = inv_transforms(image=mask_0)['image']
    mask_1_original_size = inv_transforms(image=mask_1)['image']

    current_image_has_good_features = has_good_features(image, mask_0_original_size,
                                                       area_mask_over_image_min_ratio=0.025)

    if(not current_image_has_good_features):
        return None

    file_path = os.path.join(IM_FOLDER + '/2D_slice_data/', 
                                 f'BraTS2021_{patient_id}',
                                 f'BraTS2021_{patient_id}_{mri_type}',
                                f'BraTS2021_{patient_id}_{mri_type}_{slice_index:03d}')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    mask_0_original_size *= 255  # convert to 255 scale
    mask_1_original_size *= 255
    _3channel_data = np.stack([image, mask_0_original_size, mask_1_original_size], axis=-1)

    np.save(file_path, _3channel_data)

    return int(patient_id), mri_type, slice_index, file_path+'.npy'

def sampling_one_image_update(args):
    global list_patient_id, list_mri_type, list_slice_index, list_file_path
    if(args is not None):
        patient_id, mri_type, slice_index, file_path = args
        list_patient_id.append(patient_id)
        list_mri_type.append(mri_type)
        list_slice_index.append(slice_index)
        list_file_path.append(file_path)


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
#         'MGMT_value':[row['MGMT_value']]*len(list_files),
        'mri_type':[row['mri_type']]*len(list_files),
        'file_path':list_files,
#         'fold':[row['fold']]*len(list_files)
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

#     for col_name in ['MGMT_value', 'mri_type', 'fold']:
    for col_name in ['mri_type']:
        get_first_value(df_mri_group_expand, col_name)
        
    return df_mri_group_expand
class BrainClassification2DDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv, transforms=None):
        self.csv = csv.reset_index(drop=True)
        self.augmentations = transforms
        
        if('MGMT_value' not in self.csv.columns):
            self.csv['MGMT_value'] = -1

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

def get_clf_transforms(candidate):
    dim = candidate.get('dim', DIM)
    additional_targets = {f'image{i}':'image' for i in range(SEQ_LEN-1)}
    return A.Compose(
        [
            A.augmentations.geometric.transforms.Affine(scale=1.2, always_apply=True),
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
from torch import nn

class BrainSequenceModel(nn.Module):
    def __init__(self, backbone_name, backbone_pretrained,
                 lstm_dim=64, lstm_layers=1, lstm_dropout=0., 
                 n_classes=1):
        super(BrainSequenceModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        if(backbone_pretrained is not None):
            self.backbone.load_state_dict(torch.load(backbone_pretrained))
        
        self.feature_extractor =  self.backbone.features
        self.gap = self.backbone.global_pool
        
        lstm_inp_dim = self.backbone.classifier.in_features
        
        self.lstm = nn.LSTM(lstm_inp_dim, lstm_dim, num_layers=lstm_layers, 
                            batch_first=True, bidirectional=True,
                            dropout=lstm_dropout)
        
        self.clf_head = nn.Linear(lstm_dim*2, n_classes)
        
    def forward(self, x):
        n = x.shape[0]
        seq_length = x.shape[1]
        concat_x = torch.cat([x[i] for i in range(n)], axis=0)
        concat_x = self.feature_extractor(concat_x)
        concat_x = self.gap(concat_x)
        
        stacked_x = torch.stack([concat_x[i*seq_length:i*seq_length+seq_length] for i in range(n)], axis=0)
        
        seq_features, _ = self.lstm(stacked_x)
        seq_features = seq_features[:, -1, :] # only get the last time step
        
        logits = self.clf_head(seq_features)
        
        return logits
    
class BrainSequenceModelNFNet(nn.Module):
    def __init__(self, backbone_name, backbone_pretrained,
                 lstm_dim=64, lstm_layers=1, lstm_dropout=0., 
                 n_classes=1):
        super(BrainSequenceModelNFNet, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        if(backbone_pretrained):
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
    
def predict_fn(dataloader,model, scaler, device='cuda:0'):
    '''Perform model training'''

    model.eval()
        
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    all_predictions = []
    for i, batch in tk0:

        # input, gt
        voxels, labels = batch
        voxels = voxels.to(device)
        labels = labels.to(device).float()

        # prediction
        with torch.cuda.amp.autocast(), torch.no_grad():
            logits = model(voxels)
            logits = logits.view(-1)
            probs = logits.sigmoid()
       
        all_predictions.append(probs.detach().cpu().numpy())
        
        del batch, voxels, labels, logits
        torch.cuda.empty_cache()

    all_predictions = np.concatenate(all_predictions)
    
    
    return all_predictions


seg_model = get_seg_model(SEG_MODEL)
seg_model.load_state_dict(torch.load(SEG_MODEL['model_path'], map_location='cpu'))
seg_model.to(DEVICE)

seg_model.eval()
print()


# paths = glob.glob(os.path.join(DICOM_IM_FOLDER, '00135', 'FLAIR', '*.dcm'))
# paths = sorted(paths, key=lambda x: int(x.replace('.dcm','').split("-")[-1]))

if(FAST_SUB and len(os.listdir(DICOM_IM_FOLDER))==87):
    iterations = ['00001', '00013', '00015']
else:
    iterations = os.listdir(DICOM_IM_FOLDER)

list_patient_id = []
list_slice_index = []
list_mri_type = []
list_file_path = []

os.makedirs(IM_FOLDER, exist_ok=True)
for patient_id in tqdm(iterations):
    
    s1 = time.time()
    
    all_transformed_images = []
    corresponding_mri_types = []
    all_images = []
    
    pool = Pool(processes=4)   

    for mri_type in MRI_TYPES:
        pool.apply_async(
            read_and_preprocess_voxels,
            args=(patient_id, mri_type),
            callback=read_and_preprocess_voxels_update,
            error_callback=error,
        )

    pool.close()
    pool.join()    
        
#     print(len(all_transformed_images), len(corresponding_mri_types), len(all_images))
    
    e1 = time.time()
    
    s2 = time.time()
    
    transform = get_transform(SEG_MODEL)  # transform for segmentation input
    seg_infer_ds = BrainSegmentationInferDataset(all_transformed_images, transform)
    seg_infer_loader = torch.utils.data.DataLoader(seg_infer_ds, batch_size=SEG_BATCH_SIZE, shuffle=False,
                        num_workers=N_WORKERS, pin_memory=torch.cuda.is_available())
    batch_out = batch_predict_mask(seg_infer_loader, seg_model)
    
    e2 = time.time()
    
    s3 = time.time()

    # sampling slices by mask area
    pool = Pool(processes=8)   
    
    for i in range(len(all_images)):
        image = all_images[i]
        out = batch_out[i]
        mri_type = corresponding_mri_types[i]
        
        pool.apply_async(
            sampling_one_image,
            args=(patient_id, i, image, out, mri_type),
            callback=sampling_one_image_update,
            error_callback=error,
        )

    pool.close()
    pool.join()   
    
    del batch_out
    torch.cuda.empty_cache()
        
    e3 = time.time()

    print(f'Patial time: read time: {e1-s1}. mask pred time: {e2-s2}. sampling time: {e3-s3}')



df = pd.DataFrame({
    'BraTS21ID':list_patient_id,
    'mri_type':list_mri_type,
    'slice_index':list_slice_index,
    'file_path':list_file_path,
})

df.to_csv(os.path.join(IM_FOLDER, 'meta_classification.csv'), index=False)

df_flair = df[df.mri_type=='FLAIR']
df_t1 = df[df.mri_type=='T1w']
df_t1ce = df[df.mri_type=='T1wCE']
df_t2 = df[df.mri_type=='T2w']

df_t1_group_expand = process_df_mri_type(df_t1)
df_t1ce_group_expand = process_df_mri_type(df_t1ce)
df_t2_group_expand = process_df_mri_type(df_t2)
df_flair_group_expand = process_df_mri_type(df_flair)

sub_df = []

for clf_candidate in CLF_CANDIDATES:
    mri_type = clf_candidate.get('mri_type')
    if(mri_type == 't1'):
        df_mri = df_t1_group_expand
    elif(mri_type == 't1ce'):
        df_mri = df_t1ce_group_expand
    elif(mri_type == 't2'):
        df_mri = df_t2_group_expand
    elif(mri_type == 'flair'):
        df_mri = df_flair_group_expand
    
    clf_batch_size = clf_candidate.get('batch_size', CLF_BATCH_SIZE)
    test_ds = BrainClassification2DDataset(df_mri, get_clf_transforms(clf_candidate))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=clf_batch_size, shuffle=False,
                            num_workers=N_WORKERS, pin_memory=torch.cuda.is_available())


    # Model
    if('nfnet' in clf_candidate['backbone_name'] ):
        clf_model = BrainSequenceModelNFNet(clf_candidate['backbone_name'], clf_candidate.get('backbone_pretrained'),
                                           lstm_dim=LSTM_HIDDEN_SIZE,lstm_layers=LSTM_LAYERS)
    else:
        clf_model = BrainSequenceModel(clf_candidate['backbone_name'], clf_candidate.get('backbone_pretrained'),
                                      lstm_dim=LSTM_HIDDEN_SIZE,lstm_layers=LSTM_LAYERS)
    clf_model.load_state_dict(torch.load(clf_candidate['model_path'], map_location='cpu'))
    clf_model = clf_model.to(DEVICE)
    print()
    
    scaler = torch.cuda.amp.GradScaler()
        
    test_prediction = predict_fn(test_loader, clf_model, scaler, DEVICE)
    
    tmp = df_mri.copy()
    tmp['MGMT_value'] = test_prediction

    tmp = tmp.groupby('BraTS21ID').agg({
        'MGMT_value':lambda x:x.mean()
    })
    
    sub_df.append(tmp)

sub_df = pd.concat(sub_df, axis=1).mean(axis=1).reset_index()
sub_df.columns = ['BraTS21ID', 'MGMT_value']
sub_df





