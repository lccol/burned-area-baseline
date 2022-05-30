import numpy as np
import pandas as pd

from skimage.filters import threshold_otsu
from neural_net import SatelliteDataset
from neural_net.index_functions import nbr, nbr2, bai, bais2
from neural_net.utils import compute_prec_recall_f1_acc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from pathlib import Path
from collections import defaultdict

if __name__ == '__main__':
    csv_path = Path.home() / 'datasets' / 'RESCUE' / 'sentinel-hub' / 'satellite_data.CSV'
    main_folder = Path.home() / 'datasets' / 'RESCUE' / 'sentinel-hub'
    df = pd.read_csv(csv_path)
    burned_index_str = 'nbr2'
    
    fold_def = {}
    
    for k, fold in df.groupby('fold'):
        print(f'Fold: {k}')
        fold_def[k] = set(fold['folder'].tolist())
        
    print(fold_def)
    df_dict = defaultdict(list)
    
    if burned_index_str == 'nbr2':
        burned_index_func = nbr2
        inequality = '<'
    elif burned_index_str == 'nbr':
        burned_index_func = nbr
        inequality = '<'
    elif burned_index_str == 'bai':
        burned_index_func = bai
        inequality = '>'
    elif burned_index_str == 'bais2':
        burned_index_func = bais2
        inequality = '>'
    else:
        raise ValueError(f'Burned index {burned_index_str} not yet implemented')

    dataset_config = {
        'folder': main_folder,
        'mask_intervals': [(0, 36), (37, 255)],
        'mask_one_hot': False,
        'height': 512,
        'width': 512,
        'product_list': ['sentinel2'],
        'mode': 'post',
        'filter_validity_mask': True,
        'transform': None,
        'process_dict': {
            'sentinel2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        },
        'activation_date_csv': csv_path,
        'mask_filtering': False,
        'only_burnt': True
    }
    overall_cm = np.zeros((2, 2), dtype=int)
    for fold, items in fold_def.items():
        print(f'Analying fold {fold}')
        print(f'Elements: {items}')
        
        folder_list = list(items)
        dataset = SatelliteDataset(folder_list=folder_list, **dataset_config)
        
        cm = np.zeros((2, 2), dtype=int)
        for img in dataset:
            burned_idx = burned_index_func(img['image']).squeeze()
            
            thr = threshold_otsu(burned_idx)
            if inequality == '>':
                binary = burned_idx > thr
            else:
                binary = burned_idx < thr
            
            curr_cm = confusion_matrix(img['mask'].flatten(), binary.flatten())
            cm += curr_cm
            overall_cm += curr_cm
            
            curr_prec, curr_rec, curr_f1, curr_acc = compute_prec_recall_f1_acc(curr_cm)
            
            other_acc = accuracy_score(img['mask'].flatten(), binary.flatten())
            other_prec, other_rec, other_f1, _ = precision_recall_fscore_support(img['mask'].flatten(), binary.flatten(), labels=[0, 1], pos_label=1, average='binary')
            
            assert abs(other_prec - curr_prec[1]) < 1e-4 and \
                    abs(other_rec - curr_rec[1]) < 1e-4 and \
                    abs(other_f1 - curr_f1[1]) < 1e-4 and \
                    abs(curr_acc - other_acc) < 1e-4
            
            
            
        print(cm)
        prec, rec, f1, acc = compute_prec_recall_f1_acc(cm)
        
        print(f'Precision: {prec}')
        print(f'Recall: {rec}')
        print(f'F1 score: {f1}')
        print(f'Accuracy: {acc}')
        
        df_dict['fold'].append(fold)
        df_dict['accuracy'].append(acc)
        df_dict['precision'].append(prec[1])
        df_dict['recall'].append(rec[1])
        df_dict['fscore'].append(f1[1])
        
    df = pd.DataFrame(df_dict)
    print(df)
    print('_' * 50)
    print('Mean: ')
    print(df[['accuracy', 'precision', 'recall', 'fscore']].mean())
    print('Median: ')
    print(df[['accuracy', 'precision', 'recall', 'fscore']].median())
    
    prec, rec, f1, acc = compute_prec_recall_f1_acc(overall_cm)
    print(f'Overall Precision: {prec}')
    print(f'Overall Recall: {rec}')
    print(f'Overall F1 score: {f1}')
    print(f'Overall Accuracy: {acc}')