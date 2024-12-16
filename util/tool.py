import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

# Code Reference: https://github.com/thuml/Time-Series-Library

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.lr}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


# class Transform:
#     def __call__(self, x):
#         raise NotImplementedError

#     def reverse(self, x):
#         raise NotImplementedError


# class Normalize(Transform):
#     def __init__(self, mean, var, eps=1e-5):
#         self.mean = mean
#         self.std = np.sqrt(var + eps)

#     def __call__(self, x):
#         return (x - self.mean) / self.std

#     def reverse(self, x):
#         return x * self.std + self.mean
    
def transform(x):
    last_slice = x[:, :, -1]

    # Calculate min and max along the L dimension
    min_vals = last_slice.min(dim=1, keepdim=True)[0]
    max_vals = last_slice.max(dim=1, keepdim=True)[0]  

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1 
    normalized_last_slice = (last_slice - min_vals) / range_vals

    normalized_tensor = x.clone()
    normalized_tensor[:, :, -1] = normalized_last_slice
    return min_vals, max_vals, x


def inverse_tansform(x, min_vals, max_vals):
    H = x.shape[1]
    min_vals = min_vals[:, 0].unsqueeze(1).repeat(1, H)
    max_vals = max_vals[:, 0].unsqueeze(1).repeat(1, H)
    return x * (max_vals - min_vals) + min_vals