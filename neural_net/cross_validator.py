import os
import torch
import json
import numpy as np
import pandas as pd

from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from .dataset import SatelliteDataset
from .image_processor import ProductProcessor
from .transform import *
from .index_functions import *
from .sampler import ShuffleSampler
from .stopping import EarlyStopping
from .utils import compute_prec_recall_f1_acc, compute_squared_errors, initialize_weight

from pickle import dump

from sklearn.metrics import confusion_matrix

from collections import OrderedDict, defaultdict

class CrossValidator():
    def __init__(self, groups, model_tuple: tuple, criterion_tuple: tuple, train_transforms, test_transforms, master_folder: str, csv_path: str, epochs: int, batch_size: int, lr: float, wd: float, product_list: list, mode, process_dict: dict, mask_intervals: list, mask_one_hot: bool, height: int, width: int, filter_validity_mask: bool, only_burnt: bool, mask_filtering: bool, seed: int, result_folder: str, lr_scheduler_tuple: tuple=None, ignore_list: list=None, performance_eval_func=None, squeeze_mask=True, early_stop=False, patience=None, tol=None, is_regression=False, validation_dict=None):
        self.groups = groups
        if isinstance(groups, list):
            self.groups = self._convert_list_to_dict(groups)
        assert len(model_tuple) == 2
        assert len(criterion_tuple) == 2
        assert lr_scheduler_tuple is None or len(lr_scheduler_tuple) == 2
        self.model_tuple = model_tuple
        self.criterion_tuple = criterion_tuple
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.master_folder = master_folder
        self.csv_path = csv_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.product_list = product_list
        self.mode = mode
        self.process_dict = process_dict
        self.mask_intervals = mask_intervals
        self.mask_one_hot = mask_one_hot
        self.height = height
        self.width = width
        self.filter_validity_mask = filter_validity_mask
        self.only_burnt = only_burnt
        self.mask_filtering = mask_filtering
        self.seed = seed
        self.result_folder = result_folder
        self.ignore_list = ignore_list
        self.lr_scheduler_tuple = lr_scheduler_tuple
        self.performance_eval_func = performance_eval_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.squeeze_mask = squeeze_mask
        self.early_stop = early_stop
        self.patience = patience
        self.tol = tol
        self.is_regression = is_regression
        self.validation_dict = validation_dict

        assert set(validation_dict.keys()) == set(groups.keys())

        if self.early_stop:
            assert patience is not None and tol is not None
        
        self._create_folder(self.result_folder)
        return

    @classmethod
    def _create_folder(cls, path):
        if not os.path.isdir(path):
            os.mkdir(path)
            if not os.path.isdir(path):
                raise RuntimeError('Unable to generate folder at %s' % path)

        return

    @classmethod
    def _convert_list_to_dict(cls, group_list):
        assert isinstance(group_list, list)

        result = OrderedDict()
        for idx, sublist in enumerate(group_list):
            result[idx] = sublist

        return result

    @classmethod
    def _instantiate(cls, item_tuple):
        if item_tuple is None:
            return None
        assert len(item_tuple) == 2 and isinstance(item_tuple[1], dict)
        item_class, item_args = item_tuple[0], item_tuple[1]
        return item_class(**item_args)

    def start(self, mask_postfix='mask', compute_cm=True):
        print('cuda version detected: %s' % str(torch.version.cuda))
        print('cudnn backend %s' % str(torch.backends.cudnn.version()))
        df_dict = defaultdict(list)

        ordered_keys = list(self.groups.keys())

        for idx, key in enumerate(ordered_keys):
            print('__________________________________________________________')
            print('processing fold %s' % key)
            print(self.groups[key])

            result_path = os.path.join(self.result_folder, 'fold%03d_%s' % (idx, str(key)))
            if self.validation_dict is None:
                validation_index = (idx - 1) if idx > 0 else -1
                validation_fold_name = ordered_keys[validation_index]
                validation_set = self.groups[validation_fold_name]
                print('Test set is %s, no validation dict specified... choosing %s' % (key, validation_fold_name))
            else:
                validation_fold_name = self.validation_dict[key]
                print('Test set is %s, corresponding validation set is %s' % (key, validation_fold_name))
                validation_set = self.groups[validation_fold_name]
            train_set = self._generate_train_set(validation_fold_name, key)

            cm, mse = self._start_train(train_set, validation_set, self.groups[key], result_path, str(key), mask_postfix=mask_postfix, compute_cm=compute_cm)
            df_dict['fold'].append(key)
            df_dict['test_set'].append('_'.join(self.groups[key]))

            if compute_cm:
                prec, recall, f1, acc = compute_prec_recall_f1_acc(cm)
                df_dict['accuracy'].append(acc)
                for idy in range(len(self.mask_intervals)):
                    df_dict['precision_%d' % idy].append(prec[idy])
                    df_dict['recall_%d' % idy].append(recall[idy])
                    df_dict['f1_%d' % idy].append(f1[idy])

            if self.is_regression:
                rmse = np.sqrt(mse)
                df_dict['rmse'].append(rmse[-1])
                for idy in range(len(self.mask_intervals)):
                    df_dict['rmse_%d' % idy].append(rmse[idy])

        df = pd.DataFrame(df_dict)
        df.to_csv(os.path.join(self.result_folder, 'report.csv'), index=False)
        with open(os.path.join(self.result_folder, 'test_validation_pairs.json'), 'w') as fp:
            json.dump(dict(self.groups), fp)
        return

    @classmethod
    def _get_intersect(cls, a, b):
        return set(a).intersection(set(b))

    def _start_train(self, train_set, validation_set: list, test_set: list, save_path: str, fold_key: str, mask_postfix='mask', compute_cm=True) -> np.ndarray:
        assert len(self._get_intersect(train_set, test_set)) == 0
        assert len(self._get_intersect(train_set, validation_set)) == 0
        assert len(self._get_intersect(validation_set, test_set)) == 0

        self._create_folder(save_path)

        print('Training set (%d): %s' % (len(train_set), train_set))
        print('Validation set (%d): %s' % (len(validation_set), validation_set))
        print('Test set (%d): %s' % (len(test_set), test_set))
        print('______________________________________________________________')
        print('Train - test intersect: %s' % self._get_intersect(train_set, test_set))
        print('Train - validation intersect: %s' % self._get_intersect(train_set, validation_set))
        print('Test - validation intersect: %s' % self._get_intersect(test_set, validation_set))
        print('______________________________________________________________')

        assert len(set(train_set).intersection(set(test_set))) == 0
        print('Loading train dataset')
        train_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.train_transforms, self.process_dict, self.csv_path, train_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)
        print('loading validation dataset')
        validation_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.test_transforms, self.process_dict, self.csv_path, validation_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)
        print('loading test dataset')
        test_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.test_transforms, self.process_dict, self.csv_path, test_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)

        print('Train set dim: %d' % len(train_dataset))
        print('Validation set dim: %d' % len(validation_dataset))
        print('Test set dim: %d' % len(test_dataset))

        train_sampler = ShuffleSampler(train_dataset, self.seed)
        validation_sampler = ShuffleSampler(validation_dataset, self.seed)
        test_sampler = ShuffleSampler(test_dataset, self.seed)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, drop_last=False)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, sampler=validation_sampler, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler, drop_last=False)

        print('instantiating model')
        model = self._instantiate(self.model_tuple)
        initialize_weight(model, seed=self.seed)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(self.device)
        print('model sent to device')

        criterion = self._instantiate(self.criterion_tuple)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = None
        if self.lr_scheduler_tuple is not None:
            scheduler = self.lr_scheduler_tuple[0](optimizer, **(self.lr_scheduler_tuple[1]))

        es = None
        if self.early_stop:
            es = EarlyStopping(self.patience, save_path, self.tol, verbose=True, save_best=True)

        print('starting training')
        self._train(model, criterion, optimizer, train_loader, validation_loader, self.performance_eval_func, scheduler=scheduler, early_stop=es)
        print('############### Final evaluation ###############')
        cm, mse = self._validate(model, criterion, test_loader, self.performance_eval_func, compute_results=True, compute_cm=compute_cm)

        model_path = os.path.join(save_path, fold_key + '_model.pt')
        torch.save(model.state_dict(), model_path)

        pkl_path = os.path.join(save_path, fold_key + '_dict.pkl')
        mse_values = list(range(len(self.mask_intervals)))
        mse_values.append('all')
        pkl_obj = {'cm': cm, 'train_set': train_set, 'validation_set': validation_set, 'test_set': test_set, 'mse': mse, 'mse_classes': mse_values}
        with open(pkl_path, 'wb') as f:
            dump(pkl_obj, f)

        return cm, mse

    def _generate_train_set(self, validation_fold, test_fold):
        result = []
        assert validation_fold in self.groups
        assert test_fold in self.groups
        assert validation_fold != test_fold
        assert isinstance(validation_fold, str) and isinstance(test_fold, str)
        for grp in self.groups:
            if grp == validation_fold or grp == test_fold:
                continue
            else:
                result.extend(self.groups[grp])
        return result

    def _train(self, model, criterion, optimizer, train_loader, test_loader, performance_eval_func, n_loss_print=1, scheduler=None, early_stop=None):
        for epoch in range(self.epochs):
            running_loss = 0.0
            epoch_loss = 0.0
            model.train()

            mytype = torch.float32 if isinstance(criterion, nn.MSELoss) or isinstance(criterion, nn.BCEWithLogitsLoss) else torch.long

            for idx, data in enumerate(train_loader):
                image, mask = data['image'], data['mask']

                image = image.to(self.device)
                mask = mask.to(self.device, dtype=mytype)
                if self.squeeze_mask:
                    mask = mask.squeeze(dim=1)

                optimizer.zero_grad()
                outputs = model(image)
                loss = criterion(outputs, mask)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()

                if idx % n_loss_print == (n_loss_print - 1):
                    print('[%d, %5d] loss: %f' % (epoch + 1, idx + 1, running_loss))
                    running_loss = 0.0

            val_loss = self._validate(model, criterion, test_loader, performance_eval_func, early_stop=early_stop)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if early_stop is not None and early_stop.early_stop:
                print('Terminating training...')
                break

        if early_stop is not None and (early_stop.early_stop or early_stop.best_loss < val_loss):
            print('Loading checkpoint because val_loss (%f) is higher than best_loss (%f)' % (val_loss, early_stop.best_loss))
            model.load_state_dict(torch.load(early_stop.save_path, map_location=self.device))
        return

    def _validate(self, model, criterion, loader, performance_eval_func, compute_results: bool=False, compute_cm=True, early_stop=None):
        model.eval()
        running_loss = 0.0

        if performance_eval_func is not None and hasattr(performance_eval_func, 'reset'):
            performance_eval_func.reset()

        cm = None
        mse = None
        if compute_results:
            if compute_cm:
                cm = np.zeros((len(self.mask_intervals), len(self.mask_intervals)))
            if self.is_regression:
                sq_err = np.zeros(len(self.mask_intervals) + 1)
                counters = np.zeros(len(self.mask_intervals) + 1)

        mytype = torch.float32 if isinstance(criterion, nn.MSELoss) or isinstance(criterion, nn.BCEWithLogitsLoss) else torch.long

        with torch.no_grad():
            for idx, data in enumerate(loader):
                image, mask = data['image'], data['mask']

                image = image.to(self.device)
                mask = mask.to(self.device, dtype=mytype)
                if self.squeeze_mask:
                    mask = mask.squeeze(dim=1)

                outputs = model(image)
                loss = criterion(outputs, mask)
                running_loss += loss.item()

                if performance_eval_func is not None:
                    performance_eval_func(outputs, mask)

                if compute_results:
                    if self.is_regression:
                        tmp_sq_err, tmp_counters = compute_squared_errors(outputs, mask, len(self.mask_intervals))
                        sq_err += tmp_sq_err
                        counters += tmp_counters
                        if compute_cm:
                            rounded_outputs = outputs.clamp(min=0, max=(len(self.mask_intervals) - 1)).round()
                            cm += self._compute_cm(rounded_outputs, mask)
                    else:
                        if compute_cm:
                            cm += self._compute_cm(outputs, mask)

            if performance_eval_func is not None and hasattr(performance_eval_func, 'last'):
                performance_eval_func.last()

            print('Validation running loss: %f' % running_loss)

        if early_stop is not None:
            early_stop(running_loss, model)

        if compute_results:
            if self.is_regression:
                mse = sq_err / counters
            return cm, mse
        return running_loss

    def _compute_cm(self, outputs, mask):
        keepdim = mask.shape[1] == 1
        if not self.is_regression:
            if outputs.shape[1] > 1:
                prediction = outputs.argmax(axis=1, keepdim=keepdim)
            else:
                prediction = torch.sigmoid(outputs)
                prediction = torch.where(prediction > 0.5, torch.tensor(1.0, device=outputs.device), torch.tensor(0.0, device=outputs.device))
                if not keepdim:
                    prediction = prediction.squeeze(dim=1)
        else:
            prediction = outputs
            if outputs.shape[1] == 1 and (not keepdim):
                prediction = prediction.squeeze(dim=1)
        mask = mask.cpu().numpy().flatten()
        prediction = prediction.cpu().numpy().flatten()
        
        if outputs.shape[1] == 1 and not self.is_regression:
            labels = [0, 1]
        else:
            labels = list(range(len(self.mask_intervals)))
        cm = confusion_matrix(mask, prediction, labels=labels)
        return cm