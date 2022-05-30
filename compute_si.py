import numpy as np
import pandas as pd

from neural_net import Scanner
from neural_net.index_functions import bai, bais2, nbr, nbr2

from pathlib import Path

if __name__ == '__main__':
    csv_path = Path.home() / 'datasets' / 'RESCUE' / 'sentinel-hub' / 'satellite_data.CSV'
    main_folder = Path.home() / 'datasets' / 'RESCUE' / 'sentinel-hub'
    index_func_str = 'nbr'
    if index_func_str == 'bai':
        index_func = bai
    elif index_func_str == 'bais2':
        index_func = bais2
    elif index_func_str == 'nbr':
        index_func = nbr
    elif index_func_str == 'nbr2':
        index_func = nbr2
    else:
        raise ValueError(f'not implemented yet {index_func_str}')
    
    scanner_config = {
        'folder': main_folder,
        'products': ['sentinel2'],
        'df_path': csv_path,
        'mask_intervals': [(0, 36), (37, 255)],
        'mask_one_hot': False,
        'ignore_list': None
    }
    
    df = pd.read_csv(csv_path)
    
    scanner = Scanner(**scanner_config)
    burned_acc = []
    unburned_acc = []
    for _, row in df.iterrows():
        folder = row['folder']
        img = scanner.get(folder, 'sentinel2', mode='post')
        burned_index = index_func(img).squeeze()
        mask = scanner.get_mask(folder)
        
        assert img.shape[:2] == mask.shape[:2]
        
        burned = (mask == 1).squeeze()
        burned_pixels = burned_index[burned]
        unburned_pixels = burned_index[~burned]
        
        burned_acc.append(burned_pixels)
        unburned_acc.append(unburned_pixels)
        
    burned_acc = np.concatenate(burned_acc)
    unburned_acc = np.concatenate(unburned_acc)
    
    burned_mean, burned_std = burned_acc.mean(), burned_acc.std()
    unburned_mean, unburned_std = unburned_acc.mean(), unburned_acc.std()
    
    si = (burned_mean - unburned_mean) / (burned_std + unburned_std)
    
    print(f'{index_func_str} SI: {si}')