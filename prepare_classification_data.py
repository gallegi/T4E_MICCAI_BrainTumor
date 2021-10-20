import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import json
import os
import shutil
from tqdm import tqdm
import SimpleITK as sitk

import pydicom
import glob
import sys

import time

from multiprocessing import Pool

from segmentation_models_pytorch.unetplusplus.model import UnetPlusPlus
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU

import argparse
parser = argparse.ArgumentParser(description='Insert some arguments')
parser.add_argument('--gpu', type=int,
                    help='GPU ID', default=0)
parser.add_argument('--batch_size', type=int,
                    help='Batch size', default=64)
parser.add_argument('--n_workers', type=int,
                    help='Number of parrallel workers', default=8)
args = parser.parse_args()

with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)
    
    
IM_FOLDER = SETTINGS['CLASSIFICATION_RAW_JPG']
OUT_FOLDER = SETTINGS['CLASSIFICATION_DATA_DIR']
SEGMENT_MODEL_DIR = 'models/densenet121_2d_segment/'

DEVICE = torch.device(f'cuda:{args.gpu}')

MRI_TYPES = ['T1w', 'T1wCE', 'T2w', 'FLAIR']

DIM = (224,224,3)
SEG_BATCH_SIZE = args.batch_size
N_WORKERS = args.n_workers

CANDIDATES = [
    {
        'backbone_name':'densenet121',
'model_path':f'{SEGMENT_MODEL_DIR}/Fold0_densenet121_2d_segment.pth'
    },
]

# =============== Some helper functions ================

def get_model(candidate):
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

def read_and_preprocess_voxels(patient_id, mri_type):
    paths = glob.glob(os.path.join(IM_FOLDER, patient_id, mri_type, '*.jpg'))
    paths = sorted(paths, key=lambda x: int(x.replace('.jpg','').split("-")[-1]))
    positions = []
    images = []

    for path in paths:
    #     print(path)
    #     img = pydicom.dcmread(str(dcm_path))
    #     img = img.pixel_array
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if(check_empty(img)):
            images.append(img)

    if(len(images) == 0):
        print("Found no images in case (patient_id, mri, path):", patient_id, mri_type, paths)
        return []

    voxels = np.array(images)
    voxels = normalize_voxels(voxels)  # normalize voxels to range(0,255)
#     print(len(voxels))
    return voxels, mri_type, images
        
    
def sampling_one_image(patient_id, slice_index, image, out, mri_type):

    mask_0, mask_1 = out[0], out[1]
    inv_transforms = get_inv_transform(image.shape[1], image.shape[0], candidate)
    mask_0_original_size = inv_transforms(image=mask_0)['image']
    mask_1_original_size = inv_transforms(image=mask_1)['image']

    current_image_has_good_features = has_good_features(image, mask_0_original_size, 
                                                        area_mask_over_image_min_ratio=0.025)

    if(not current_image_has_good_features):
        return None

    file_path = os.path.join(OUT_FOLDER + '/2D_slice_data/', 
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
# =======================================================        
        
    
# =============== Generate masks combined with image ==============
if(os.path.exists(OUT_FOLDER)):
    shutil.rmtree(OUT_FOLDER)  # REMOVE EXISTING DIR. BECARE FULL USING THIS        
               
candidate = CANDIDATES[0]
model = get_model(candidate)
model.load_state_dict(torch.load(candidate['model_path']))
model.to(DEVICE)

model.eval()
print()        
        
list_patient_id = []
list_slice_index = []
list_mri_type = []
list_file_path = []

os.makedirs(OUT_FOLDER, exist_ok=True)
for pi, patient_id in tqdm(enumerate(os.listdir(IM_FOLDER))):

    s1 = time.time()
    
    all_transformed_images = []
    corresponding_mri_types = []
    all_images = []
    
    pool = Pool(processes=N_WORKERS)   

    for mri_type in MRI_TYPES:
        pool.apply_async(
            read_and_preprocess_voxels,
            args=(patient_id, mri_type),
            callback=read_and_preprocess_voxels_update,
            error_callback=error,
        )

    pool.close()
    pool.join()    
            
    e1 = time.time()
    
    s2 = time.time()
    
    transform = get_transform(candidate)  # transform for segmentation input
    seg_infer_ds = BrainSegmentationInferDataset(all_transformed_images, transform)
    seg_infer_loader = torch.utils.data.DataLoader(seg_infer_ds, batch_size=SEG_BATCH_SIZE, shuffle=False,
                        num_workers=N_WORKERS, pin_memory=torch.cuda.is_available())
    batch_out = batch_predict_mask(seg_infer_loader, model)
    
    e2 = time.time()
    
    s3 = time.time()

    # sampling slices by mask area
    pool = Pool(processes=N_WORKERS)   
    
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

#     print(f'Patial time: read time: {e1-s1}. mask pred time: {e2-s2}. sampling time: {e3-s3}')
        
out_df = pd.DataFrame({
    'BraTS21ID':list_patient_id,
    'mri_type':list_mri_type,
    'slice_index':list_slice_index,
    'file_path':list_file_path,
})

out_df.to_csv(os.path.join(OUT_FOLDER, 'meta_classification.csv'), index=False)

# ===========================================================
