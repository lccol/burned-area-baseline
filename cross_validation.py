import os
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms

from neural_net import ProductProcessor
from neural_net.unet import UNet
from neural_net.cross_validator import CrossValidator
from neural_net.transform import *
from neural_net.loss import *
from neural_net.performance_storage import *
from neural_net.utils import set_seed
from neural_net.index_functions import nbr

from collections import OrderedDict
from pathlib import Path

def main():
    seed = 47
    set_seed(seed)

    epochs = 50
    batch_size = 8
    wd = 0

    validation_dict = {'purple': 'coral',
                   'coral': 'cyan',
                   'pink': 'coral',
                   'grey': 'coral',
                   'cyan': 'coral',
                   'lime': 'coral',
                   'magenta': 'coral'
                  }

    all_bands_selector = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    print('cuda version detected: %s' % str(torch.version.cuda))
    print('cudnn backend %s' % str(torch.backends.cudnn.version()))

    
    base_result_path = Path.home() / 'logs' / 'burned_area_dataset_paper'
    if not base_result_path.is_dir():
        base_result_path.mkdir(parents=True)
    fold_definition = Path.home() / 'datasets' / 'burned_area_dataset' / 'sentinel-hub' / 'satellite_data.CSV'

    master_folder = Path.home() / 'datasets' / 'burned_area_dataset' / 'sentinel-hub'
    csv_path = fold_definition
    # n_classes = 1 #len(mask_intervals)
    mask_one_hot = False
    only_burnt = True
    mask_filtering = False
    filter_validity_mask = True
    patience = 5
    tol = 1e-2
    height, width = 512, 512

    groups = OrderedDict()
    df = pd.read_csv(fold_definition)
    grpby = df.groupby('fold')
    for grp in grpby:
        folder_list = grp[1]['folder'].tolist()

        print('______________________________________')
        print('fold key: %s' % grp[0])
        print('folders (%d): %s' % (len(folder_list), str(folder_list)))
        groups[grp[0]] = folder_list

    if not os.path.isdir(base_result_path):
        raise RuntimeError('Invalid base result path %s' % base_result_path)
        
    
    print('##############################################################')
    print('RESULT PATH: %s' % result_path)
    print('##############################################################')
    
    result_path = base_result_path / 'binary_unet_dice'
    lr = 1e-4
    mask_intervals = [(0, 36), (37, 255)]
    product_list = ['sentinel2']
    mode = 'post'
    process_dict = {
        'sentinel2': all_bands_selector,
    }
    n_channels = 12

    train_transform = transforms.Compose([
        RandomRotate(0.5, 50, seed=seed),
        RandomVerticalFlip(0.5, seed=seed),
        RandomHorizontalFlip(0.5, seed=seed),
        RandomShear(0.5, 20, seed=seed),
        ToTensor(round_mask=True),
        Normalize((0.5,) * n_channels, (0.5,) * n_channels)
    ])

    test_transform = transforms.Compose([
        ToTensor(round_mask=True),
        Normalize((0.5, ) * n_channels, (0.5, ) * n_channels)
    ])

    print('#' * 50)
    print('####################### CV all post binary UNET with DiceLoss #######################')
    print('RESULT PATH: %s' % result_path)
    print('BATCH SIZE: %d' % batch_size)
    print('#' * 50)

    model_class = UNet
    model_args = {'n_classes': 2, 'n_channels': n_channels, 'act': 'relu'}
    model_tuple = (model_class, model_args)

    loss_class = GDiceLossV2
    loss_args = {'apply_nonlin': nn.Softmax(dim=1), 'self_compute_weight': True}
    criterion_tuple = (loss_class, loss_args)
    performance_evaluator = AccuracyMulticlassStorage(one_hot=mask_one_hot)

    scheduler_tuple = None

    cv = CrossValidator(groups, model_tuple, criterion_tuple, train_transform, test_transform, master_folder, csv_path, epochs, batch_size, lr, wd, product_list, mode, process_dict, mask_intervals, mask_one_hot, height, width, filter_validity_mask, only_burnt, mask_filtering, seed, result_path, performance_eval_func=performance_evaluator, squeeze_mask=False, early_stop=True, patience=patience, tol=tol, lr_scheduler_tuple=scheduler_tuple, is_regression=False, validation_dict=validation_dict)
    cv.start()

    return

if __name__ == '__main__':
    main()