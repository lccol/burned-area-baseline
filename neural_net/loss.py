import torch
import numpy as np

from torch import nn
from torch.autograd import Variable

class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5, weight=None, compact_data=True, self_compute_weight=False):
        """
        Generalized Dice;
        Credits to: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.class_weight = weight
        self.compact_data = compact_data
        self.self_compute_weight = self_compute_weight

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, value=1)


        nonlin_output = net_output
        if self.apply_nonlin is not None:
            nonlin_output = self.apply_nonlin(nonlin_output)

        my_in = self.flatten(nonlin_output)
        target = self.flatten(y_onehot)
        target = target.float()
        if self.self_compute_weight:
            target_sum = target.sum(-1)
            class_weights = 1. / (target_sum * target_sum).clamp(min=self.smooth)
            class_weights = class_weights.detach()

        if self.self_compute_weight:
            intersect = (my_in * target).sum(-1) * class_weights
        else:
            intersect = (my_in * target).sum(-1)
        if self.class_weight is not None:
            weight = self.class_weight.detach()
            intersect = weight * intersect
        if self.compact_data:
            intersect = intersect.sum()

        if self.self_compute_weight:
            denominator = ((my_in + target).sum(-1) * class_weights).sum()
        else:
            denominator = (my_in + target).sum(-1)
        if self.compact_data:
            denominator = denominator.sum()

        result = 1. - 2. * intersect / denominator.clamp(min=self.smooth)
        return result

    @classmethod
    def flatten(cls, tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
        (N, C, D, H, W) -> (C, N * D * H * W)
        """
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(C, -1)

class IndexLoss(nn.Module):
    def __init__(self, index, gt_one_hot, loss, device=None):
        super(IndexLoss, self).__init__()
        assert index == 0 or index == 1
        self.index = index
        self.gt_one_hot = gt_one_hot
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = loss

    def forward(self, outputs, gt):
        out = outputs[self.index]
        if self.index == 0:
            bin_gt = gt
            if self.gt_one_hot:
                bin_gt = gt.argmax(dim=1, keepdim=True)
            bin_gt = torch.where(bin_gt >= 1.0, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))

            return self.loss(out, bin_gt)
        elif self.index == 1:
            return self.loss(out, gt)

class MyMSE(nn.Module):
    def __init__(self, n_classes, smooth=1e-5):
        super(MyMSE, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        return
    
    def forward(self, outputs, gt):
        gt_scattered = self.scatter(gt)
        out_scattered = self.scatter(outputs, gt)

        elem_counters = self.count_elements(gt).clamp(min=self.smooth)
        diff = gt_scattered - out_scattered
        diff = (diff ** 2).sum((0, *range(2, len(diff.shape))))
        mse = diff / elem_counters
        return  mse

    def count_elements(self, gt):
        with torch.no_grad():
            shp = (gt.shape[0], self.n_classes, *(gt.shape[2:]))
            counter = torch.zeros(shp, device=gt.device)
            gt = gt.long()
            counter.scatter_(1, gt, value=1.)
            result = counter.sum((0, *(range(2, len(gt.shape)))))
        return result

    def scatter(self, t, index=None):
        assert t.shape[1] == 1
        shp = (t.shape[0], self.n_classes, *(t.shape[2:]))
        result = torch.zeros(shp, device=t.device)
        if index is None:
            index = t
        index = index.long()
        t = t.float()
        result.scatter_(1, index, t)
        return result