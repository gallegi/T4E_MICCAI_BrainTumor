import pandas as pd
import numpy as np
import json
import re
import torch
import random
import sys
import os
import matplotlib.pyplot as plt
import logging
sys.path.append('/tf/show-us-the-data/utils/')

from transformers import AutoTokenizer
from tqdm import tqdm
from coleridge_datasets import ColeridgePredictDataset
from nltk import sent_tokenize
from copy import deepcopy
from typing import List

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(name, path, mode='a'):
    logger = logging.getLogger(name)  

    if not logger.hasHandlers():
        # set log level
        logger.setLevel(logging.INFO)

        # define file handler and set formatter
        file_handler = logging.FileHandler(path, mode=mode)
        formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formatter)

        # add file handler to logger
        logger.addHandler(file_handler)
    
    return logger
    
    
def log_and_print(logger, obj):
    print(obj)
    logger.info(obj)    
    
def init_progress_dict(metrics):
    progress_dict = dict()
    for metric in metrics:
        progress_dict[f'train_{metric}'] = []
        progress_dict[f'valid_{metric}'] = []
    return progress_dict

def log_to_progress_dict(progress_dict, metric_dict):
    for k, v in metric_dict.items():
        progress_dict[k].append(v)
       
    return progress_dict

def save_progress(progress_dict, out_folder, out_folder_name, fold, show=False):
    metric_names = list(progress_dict.keys())
    epochs = len(progress_dict[metric_names[0]])+1
    
    # plot figure and save the progress chart
    n_cols = 4
    n_rows = int(np.ceil(len(metric_names) / 2 / n_cols))
    
    plt.figure(figsize=(7*n_cols, 7*n_rows))
    
    for i in range(0, len(metric_names), 2):
        plt.subplot(n_rows,n_cols,i/2+1)

        plt.plot(range(1, epochs), progress_dict[metric_names[i]])
        plt.plot(range(1, epochs), progress_dict[metric_names[i+1]])
        plt.legend([metric_names[i], metric_names[i+1]])
        plt.xlabel('Epoch')
        plt.title(f'{metric_names[i]} and {metric_names[i+1]}')

    save_name = f'training_progress_{out_folder_name}_fold{fold}'
    plt.savefig(os.path.join(out_folder, save_name+'.jpg'))

    if(show):
        plt.show()

    pd.DataFrame({'epoch':range(1, epochs), **progress_dict}).to_csv(os.path.join(out_folder, save_name+'.csv'), index=False)

    
def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x