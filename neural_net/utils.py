import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from torch import nn

def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def produce_report(cm, result_path, test_set=None, sq_err=None, num_el=None, cm2=None):
    def _write_cm(f, cm):
        f.write('__________________________________________________________________________________________\n')
        f.write('Labels order (left-to-right, top-to-bottom): %s\n' % list(range(cm.shape[0])))
        f.write('__________________________________________________________________________________________\n')
        f.write('Confusion matrix:\n')
        f.write(str(cm) + '\n')
        f.write('Rows = ground truth\n')
        f.write('Columns = prediction\n')
        f.write('__________________________________________________________________________________________\n')
        total_per_class = cm.sum(axis=1)
        total = cm.sum()
        for x in range(cm.shape[0]):
            f.write('Total number of pixels in class %d: %d - %.3f (percentage)\n' % (x, total_per_class[x], (total_per_class[x] / total)))
        f.write('__________________________________________________________________________________________\n')
        f.write('Performances:\n')
        prec, rec, f1, acc = compute_prec_recall_f1_acc(cm)
        f.write('Overall accuracy: %.4f\n' % acc)
        f.write('###############\n')
        for x in range(cm.shape[0]):
            f.write('Class %d:\n' % x)
            f.write('Precision: %.4f\n' % prec[x])
            f.write('Recall: %.4f\n' % rec[x])
            f.write('f1: %.4f\n' % f1[x])
            f.write('###############\n')

        return

    with open(result_path, 'w') as f:
        if test_set is not None:
            f.write('Test set:')
            for el in test_set:
                f.write(el + ', ')
            f.write('\n')

        f.write('Confusion matrix #1\n')
        _write_cm(f, cm)
        if cm2 is not None:
            f.write('Confusion matrix #2\n')
            _write_cm(f, cm2)

        if sq_err is not None and num_el is not None:
            results = np.sqrt(sq_err / num_el)
            f.write('__________________________________________________________________________________________\n')
            f.write('RMSE on entire test set: %f\n' % results[-1])
            f.write('###############\n')
            for x in range(sq_err.size - 1):
                f.write('Class %d:\n' % x)
                f.write('RMSE: %f\n' % results[x])
                f.write('###############\n')

def compute_squared_errors(prediction, ground_truth, n_classes, check=True):
    squared_errors = []
    counters = []

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().cpu().numpy()

    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.squeeze().cpu().numpy()

    if len(ground_truth.shape) == 3 and ground_truth.shape[-1] == 1:
        ground_truth = ground_truth.squeeze(axis=-1)

    if len(prediction.shape) == 3 and prediction.shape[-1] == 1:
        prediction = prediction.squeeze(axis=-1)

    mse_check = []

    for idx in range(n_classes):
        mask = ground_truth == idx
        pred_data = prediction[mask]
        gt_data = ground_truth[mask]
        sq_err = np.square(pred_data - gt_data).sum()
        n_elem = mask.sum()

        squared_errors.append(sq_err)
        counters.append(n_elem)

        if check:
            if n_elem > 0:
                mse_check.append(mean_squared_error(gt_data, pred_data))
            else:
                mse_check.append(0)

    sq_err = np.square((prediction - ground_truth).flatten()).sum()
    if check:
        mse_check.append(mean_squared_error(ground_truth.flatten(), prediction.flatten()))
    n_elem = prediction.size
    squared_errors.append(sq_err)
    counters.append(n_elem)

    sq_err_res = np.array(squared_errors)
    count_res = np.array(counters)

    if check:
        mymse = sq_err_res / count_res.clip(min=1e-5)
        mymse[np.isnan(mymse)] = 0

        mse_check = np.array(mse_check)
        assert (np.abs(mymse - mse_check) < 1e-6).all()

    return sq_err_res, count_res

def compute_prec_recall_f1_acc(conf_matr):
    accuracy = np.trace(conf_matr) / conf_matr.sum()

    predicted_sum = conf_matr.sum(axis=0)
    gt_sum = conf_matr.sum(axis=1)
                
    diag = np.diag(conf_matr)
    precision = diag / predicted_sum.clip(min=1e-5)
    recall = diag / gt_sum.clip(min=1e-5)
    f1 = 2 * (precision * recall) / (precision + recall).clip(min=1e-5)
    return precision, recall, f1, accuracy

def initialize_weight(model, seed=None):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d) or isinstance(model, nn.Linear):
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.xavier_normal_(model.weight.data)
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.normal_(model.bias.data)