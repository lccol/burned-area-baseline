import os
import re
import numpy as np
import pandas as pd

from .scanner import Scanner
from .image_processor import ImageProcessor, ProductProcessor
from torch.utils.data.dataset import Dataset

from collections import OrderedDict

class SatelliteDataset(Dataset):
    """
    SatelliteDataset class used to load all the images from a master folder with all the specified products and modes (i.e. 'pre' and 'post') with the corresponding masks.
    All the images are processed using ImageProcessor, ProductProcessor to eventually generate intermediate products such as NBR2, BAIS2 and other indexes and then cut to a specified shape using a tile strategy.
    """
    def __init__(self, folder, mask_intervals: list, mask_one_hot: bool, height: int, width: int, product_list: list, mode, filter_validity_mask, transform=None, process_dict: dict=None, activation_date_csv: str=None, folder_list: list=None, ignore_list: list=None, mask_filtering: bool=False, only_burnt: bool=False, mask_postfix: str='mask'):
        """
        Constructor of SatelliteDataset. Extends pytorch Dataset class.

        Args:
            folder (str): path to master folder containing different subfolder, one for each area of interest
            mask_intervals (list(tuple(int, int))): set of intervals used to discretize grayscale masks into levels. Each tuple specifies the range of pixel values (limits are included) which are associated to a class. e.g. (0, 63) -> all the pixels with values between 0 and 63 included are associated to idx, where idx is the index given by enumerate(mask_intervals).
            mask_one_hot (bool): flag to specify whether masks must be returned with one-hot-encoding or a 1-channel image
            height (int): height used to cut images
            width (int): width used to cut images
            product_list (list(str)): list of product names used to collect data from subfolders e.g. ['sentinel2', 'sentinel1', 'dem', 'cloud_coverage']
            mode (Union[dict, str]): either a string among ['both', 'pre', 'post'] or a dict. If it is a string, all the products are collected with the specified mode. If it is a dict, each product (= key) must be specified with a list containing 'pre', 'post' or 'both' to collect information. List are order-sensitive: if 'pre' is specified before 'post', pre-fire images are collected before 'post'
            transform: PyTorch transformers
            filter_validity_mask (Union[bool, list(str)]): specifies whether or not the last channel must be used as a validity mask to zero all the values for which validity_mask == 0 in all the other channels. If bool, the operation is performed to all the products. If list, the operation is performed only to the selected products.
            process_dict (dict): dictionary used to specify how to process and elaborate each product. Key is product name (e.g. 'sentinel2'). Value must be either a list(int) used to select only specific channels on both pre-fire and post-fire images or an instance of ProductProcessor class to process pre-fire and post-fire images. If None, pre-fire and post-fire images are concatenated along the channel axis, in case both are present
            activation_date_csv (str): path to a csv containing two columns: folder and activation_date, which associates to each subfolder the activation date of the event
            folder_list (list(str)): list of subfolders to be analyzed. If None, all the subfolders in self.folder are analyzed
            ignore_list (list(str)): list of subfolders to be ignored. If None, all the subfolders are considered
            mask_filtering (bool): if True, use the mask to put all the non-burnt values of the image to 0, for all the products and their channels. Default is False
            only_burnt (bool): if True, filter only the images with at least 1 burnt pixel and discard areas with no burned regions. Default is false
            mask_postfix (str): mask postfix used to retrieve mask. By default, in each subfolder the mask is retrieved with the filename 'subfolder_mask.tiff'.
        """
        self.folder = folder
        self.mask_intervals = mask_intervals
        self.mask_one_hot = mask_one_hot
        self.height = height
        self.width = width
        self.product_list = product_list
        self.mode = mode
        self.filter_validity_mask = filter_validity_mask
        self.mask_filtering = mask_filtering
        self.only_burnt = only_burnt
        self.mask_postfix = mask_postfix
        self.apply_mask_discretization = True
        if self.mask_postfix != 'mask':
            self.apply_mask_discretization = False

        if isinstance(filter_validity_mask, bool):
            self.filter_validity_flag = True
        elif isinstance(filter_validity_mask, list) or isinstance(filter_validity_mask, set):
            self.filter_validity_flag = True
        elif filter_validity_mask is None:
            self.filter_validity_flag = False
        else:
            raise ValueError('Invalid value for filter_validity_mask %s' % str(filter_validity_mask))

        self.transform = transform
        self.process_dict = process_dict
        self.activation_date_csv = activation_date_csv

        self.folder_list = folder_list
        self.ignore_list = ignore_list

        if self.folder_list is None:
            self.folder_list = []
            for dirname in os.listdir(self.folder):
                if self.ignore_list is not None and dirname in self.ignore_list:
                    continue

                self.folder_list.append(dirname)

        self.scanner = Scanner(folder, product_list, activation_date_csv, mask_intervals=mask_intervals, mask_one_hot=mask_one_hot, ignore_list=self.ignore_list, valid_list=self.folder_list)
        self.processor = ImageProcessor(height, width)

        regexp = r'^(((?!pre|post).)+)(_(pre|post))?$'
        regexp = re.compile(regexp, re.IGNORECASE)

        self.images = []
        self.masks = []
        for idx, dirname in enumerate(self.folder_list):
            image, product = self.scanner.get_all(dirname, self.product_list, mode=mode, retrieve_mask=True, mask_postfix=self.mask_postfix, apply_mask_discretization=self.apply_mask_discretization)
            brnt_mask = self.scanner.get_mask(dirname)
            assert product[-1] == 'mask'

            # skip images at full resolution which do not contain any burned area
            if self.only_burnt:
                tmp_mask = brnt_mask
                if self.mask_one_hot:
                    tmp_mask = tmp_mask.argmax(axis=-1, keep_dims=True)
                if not ((tmp_mask >= 1).any()):
                    continue
                
            if self.filter_validity_flag:
                for img, prod in zip(image, product):
                    search_res = regexp.search(prod)
                    if not search_res:
                        raise ValueError('Invalid product name encountered %s' % prod)
                    base_prod = search_res.group(1)
                    if prod == 'mask':
                        continue
                    if (isinstance(self.filter_validity_mask, bool) and self.filter_validity_mask) or ((isinstance(self.filter_validity_mask, list) or isinstance(self.filter_validity_mask, set)) and base_prod in self.filter_validity_mask):
                        bool_mask = img[:, :, -1] == 0
                        img[:, :, :-1][bool_mask] = 0

            channel_counter = [x.shape[-1] for x in image]
            image = self.processor.upscale(image, product, dirname, concatenate=False)
            brnt_mask = self.processor.upscale([brnt_mask], ['mask'], dirname, concatenate=False)
            if process_dict is not None:
                image, product = self.processor.process(image, product, process_dict, return_ndarray=True, channel_counter=channel_counter)

            if self.mask_filtering:
                tmp_mask = brnt_mask[0]
                nonburned_mask = tmp_mask == 0 if not self.mask_one_hot else tmp_mask[..., 0] == 1
                if len(nonburned_mask.shape) == 3:
                    nonburned_mask = nonburned_mask.squeeze(axis=-1)
                if isinstance(image, np.ndarray):
                    image[..., :-1][nonburned_mask] = 0
                elif isinstance(image, list):
                    for idx, tmp_img in enumerate(image):
                        if idx != (len(image) - 1):
                            image[idx][nonburned_mask] = 0
                else:
                    raise ValueError('Invalid image type %s' % str(type(image)))

            image, count = self.processor.cut(image, product, return_ndarray=True, apply_mask_round=self.apply_mask_discretization)
            image, mask = image[..., :-1], image[..., -1:]

            brnt_mask, _ = self.processor.cut(brnt_mask, ['mask'], return_ndarray=True, apply_mask_round=True)

            # second filtering operation: exclude all the smaller portion of images which do not contain any burned region
            if self.only_burnt:
                tmp_mask = brnt_mask
                if self.mask_one_hot:
                    tmp_mask = tmp_mask.argmax(axis=-1, keep_dims=True)
                valid_cut = (tmp_mask > 0).any(axis=(1, 2))
                valid_cut = np.arange(valid_cut.shape[0])[valid_cut.flatten()]
                image = image[valid_cut, ...]
                mask = mask[valid_cut, ...]

            self.images.append(image)
            self.masks.append(mask)

        assert len(self.images) == len(self.masks)

        self.images = np.concatenate(self.images, axis=0)
        self.masks = np.concatenate(self.masks, axis=0)

        assert self.images.shape[0] == self.masks.shape[0]
        return

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return self.images.shape[0]

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.

        Returns:
            dict(np.ndarray): returns a dict containing {'image': image_np.ndarray, 'mask': mask_np.ndarray}
        """
        result = {}
        result['image'] = self.images[idx]
        result['mask'] = self.masks[idx]

        if self.transform is not None:
            result = self.transform(result)
        return result