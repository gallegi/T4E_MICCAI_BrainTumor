import os
import json

import logging
import pandas as pd 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from monai.transforms import (
    AddChannel,
    Compose,
    Resize,
    Transform
)
from monai.data import DataLoader, ImageDataset

import torch.nn.functional as F

from multiprocessing import Pool

with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)

IM_FOLDER_TASK1 = SETTINGS['TASK1_DIR']

RUN_FOLDS = [0]
KFOLD_PATH = SETTINGS['KFOLD_PATH']

SEED = 67
N_PROCESSES = 8

OUT_FOLDER = SETTINGS['SEGMENT_DATA_DIR']

PLANES = ['sagital', 'coronal', 'axial']
MRI_TYPES = ['t1', 't1ce', 't2', 'flair']

# ============ Helper functions ===========
class ScaleRange(Transform):
    def __init__(self, new_max = 255.0):
        super(ScaleRange, self).__init__()
        self.new_max = new_max
        
    def __call__(self, data):
        dmin, dmax = data.min(), data.max()
        return (data - dmin) / (dmax-dmin) * self.new_max

class ConvertToMultiChannelBasedOnBratsClasses(Transform):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    Ehancing Tumor (ET) = enhancing tumor
    Tumor Core (TC) = enhancing tumor + necrotic
    Whole Tumor (WT) = enhancing tumor + necrotic + edema    
    """

    def __call__(self, masks):
        '''This time we only use 2 label: 0 - WT and 1 - ET'''
        result = []

        # merge labels 1, 2 and 4 to construct WT
        result.append(
            np.logical_or(
                np.logical_or(masks == 1, masks == 2), masks == 4
            )
        )
        # label 4 is ET
        result.append(masks == 4)
        
        return np.stack(result, axis=0).astype(np.float32)

def get_non_0_voxels_and_masks(voxels, masks_2channels, ax=0, min_avg=0.01):
    '''Get non-empty slices from the 3D mask
        A 2D slice is considered to be empty if its mean pixel value < min_avg'''
    masks = np.logical_or(masks_2channels[0], masks_2channels[1])
    remain_axes = tuple([i for i in range(len(voxels.shape)) if i != ax])
    ax_mean = masks.mean(axis=remain_axes)
    ax_non_0_inds = ax_mean > min_avg
    if(ax==0):
        return voxels[ax_non_0_inds], masks_2channels[:, ax_non_0_inds, :, :]
    if(ax==1):
        return voxels[:,ax_non_0_inds,:], masks_2channels[:, :, ax_non_0_inds,:]
    if(ax==2):
        return voxels[:,:,ax_non_0_inds], masks_2channels[:,:,:,ax_non_0_inds]
    
def sampling_slices(non_0_voxels, non_0_masks, ax=0, keep_rate=0.1):
    '''Nearby slices are similar to each other, we use sample to only get the different ones'''
    total_slices = non_0_voxels.shape[ax]
    T = max(round(total_slices * keep_rate), 1)
    sampling_inds = np.arange(0, total_slices, T)
    
    if(ax==0):
        return non_0_voxels[sampling_inds], non_0_masks[:, sampling_inds, :, :]
    if(ax==1):
        return non_0_voxels[:, sampling_inds, :], non_0_masks[:, :, sampling_inds, :]
    if(ax==2):
        return non_0_voxels[:, :, sampling_inds], non_0_masks[:, :, :, sampling_inds]
    
    
def process_one_patient(voxels, masks, patient_id):
    '''Perform slicing 2D images and tumor masks for this patient'''
    current_list_patient_id = []
    current_list_plane = []
    current_list_mri_type = []
    current_list_slice_index = []
    current_list_file_path = []
    current_list_segfile_path = []
    
    for ax, plane in enumerate(PLANES):
        non_0_voxels, non_0_masks = get_non_0_voxels_and_masks(voxels, masks, ax=ax)
        if(non_0_voxels.shape[ax]==0):
            print(f'Cannot get any slice in patient: {patient_id}, plane: {plane} due to the masks are to small')
            continue
        sampled_non_0_voxels, sampled_non_0_masks = sampling_slices(non_0_voxels, non_0_masks, ax=ax)

        for j in range(sampled_non_0_voxels.shape[ax]):
            file_path = os.path.join(OUT_FOLDER + '/2D_slice_data/', 
                                     f'BraTS2021_{patient_id:05d}',
                                     f'BraTS2021_{patient_id:05d}_{mri_type}',
                                    f'BraTS2021_{patient_id:05d}_{mri_type}_{plane}_{j:03d}')
            seg_file_path = os.path.join(OUT_FOLDER + '/2D_slice_data/', 
                                     f'BraTS2021_{patient_id:05d}',
                                    f'BraTS2021_{patient_id:05d}_segmask',
                                    f'BraTS2021_{patient_id:05d}_segmask_{plane}_{j:03d}')

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            os.makedirs(os.path.dirname(seg_file_path), exist_ok=True)
            
            if(ax==0):
                np.save(file_path, sampled_non_0_voxels[j])
                np.save(seg_file_path, sampled_non_0_masks[:,j])
            elif(ax==1):
                np.save(file_path, sampled_non_0_voxels[:,j,:])
                np.save(seg_file_path, sampled_non_0_masks[:,:,j,:])
            elif(ax==2):
                np.save(file_path, sampled_non_0_voxels[:,:,j])
                np.save(seg_file_path, sampled_non_0_masks[:,:,:,j])
            else:
                raise ValueError('No such ax')

            current_list_patient_id.append(patient_id)
            current_list_plane.append(plane)
            current_list_mri_type.append(mri_type)
            current_list_slice_index.append(j)
            current_list_file_path.append(file_path)
            current_list_segfile_path.append(seg_file_path)

    return current_list_patient_id, current_list_plane, current_list_mri_type,  \
            current_list_slice_index, current_list_file_path, current_list_segfile_path


def update(args):
    global list_patient_id, list_plane, list_mri_type, list_slice_index, list_file_path, list_segfile_path
    pbar.update()
    current_list_patient_id, current_list_plane, current_list_mri_type,  \
            current_list_slice_index, current_list_file_path, current_list_segfile_path = args
    
    list_patient_id += current_list_patient_id
    list_plane += current_list_plane
    list_mri_type += current_list_mri_type
    list_slice_index += current_list_slice_index
    list_file_path += current_list_file_path
    list_segfile_path += current_list_segfile_path


def error(e):
    print(e)
        
# =========================================

# ============ Read meta data =============
fold_df = pd.read_csv(KFOLD_PATH)
fold_df['pfolder'] = fold_df.BraTS21ID.map(lambda x: f'BraTS2021_{x:05d}')

df = pd.DataFrame(os.listdir(IM_FOLDER_TASK1), columns=['pfolder'])

print(df)

df['BraTS21ID'] = df['pfolder'].map(lambda x: int(x.split('_')[-1]) if x.startswith('BraTS2021_') else np.nan)

df = df[~df.BraTS21ID.isin(fold_df.BraTS21ID.tolist())]

for t in MRI_TYPES:
    df[f'{t}_data_path'] = df.pfolder.map(lambda x: os.path.join(IM_FOLDER_TASK1, x, x+f'_{t}.nii.gz'))
df['seg_label_path'] = df.pfolder.map(lambda x: os.path.join(IM_FOLDER_TASK1, x, x+f'_seg.nii.gz'))

# =========================================


# ============ Create a nii gz file loader ==========
transforms = Compose([ScaleRange()])

seg_transforms = Compose([ConvertToMultiChannelBasedOnBratsClasses(),
                         ])

mri_type = MRI_TYPES[0]
# Define nifti dataset, data loader
dataset = ImageDataset(image_files=df[f'{mri_type}_data_path'].tolist(),
                             seg_files = df.seg_label_path.tolist(),
                             seg_transform=seg_transforms,
                            transform=transforms
                      )
# =====================================================



# ========== Perform slicing data and mask ============

for mri_type in MRI_TYPES:
    dataset = ImageDataset(image_files=df[f'{mri_type}_data_path'].tolist(),
                                 seg_files = df.seg_label_path.tolist(),
                                   labels = df['BraTS21ID'].tolist(),
                                 seg_transform=seg_transforms,
                                transform=transforms
                          )
    
    os.makedirs(OUT_FOLDER + '/2D_slice_data/', exist_ok=True)

    list_patient_id = []
    list_plane = []
    list_mri_type = []
    list_slice_index = []
    list_file_path = []
    list_segfile_path = []

    pool = Pool(processes=N_PROCESSES)   

    iterations = range(len(dataset))
    pbar = tqdm(iterations)

    for i in iterations:
        voxels, masks, patient_id = dataset[i]
        pool.apply_async(
            process_one_patient,
            args=(voxels, masks, patient_id),
            callback=update,
            error_callback=error,
        )

    pool.close()
    pool.join()
    pbar.close()
    
out_df = pd.DataFrame({
    'BraTS21ID':list_patient_id,
    'mri_type':list_mri_type,
    'plane':list_plane,
    'slice_index':list_slice_index,
    'file_path':list_file_path,
    'segfile_path':list_segfile_path
})

out_df.to_csv(os.path.join(OUT_FOLDER, 'segment_meta.csv'))
# =====================================================


# ================= Kfold split ====================
kfold = GroupKFold(n_splits=5)

i = 0
for train_ind, valid_ind in kfold.split(df,df,df['BraTS21ID']):
    df.loc[valid_ind, 'fold'] = i
    i+=1
df.to_csv(f'{OUT_FOLDER}/segment_meta_groupkfold.csv', index=False)
# ==================================================
    
    
  