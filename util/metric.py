import numpy as np
import torch
import torch.nn as nn

def quantile_loss(a, b, q):
    # a: prediction, b: ground truth
    assert 0 <= q <= 1
    loss = np.where(b >= a, q * (b - a), (1 - q) * (a - b))
    return loss

# def quantile_loss_tensor(predictions, targets, q):
#     assert 0 <= q <= 1, "Quantile level q must be between 0 and 1"
    
#     loss = torch.where(targets >= predictions, 
#                        q * (targets - predictions), 
#                        (1 - q) * (predictions - targets))
#     return loss.mean() 


class quantile_loss_tensor(nn.Module):
    def __init__(self, q):
        super(quantile_loss_tensor, self).__init__()    
        self.q = q

    def forward(self, predictions, targets):
        loss = torch.where(targets >= predictions, 
                        self.q * (targets - predictions), 
                        (1.0 - self.q) * (predictions - targets))
        return loss.mean() 


def root_mean_square_error(a, b):
    # a: prediction, b: ground truth
    return np.sqrt(np.mean((a - b) ** 2))


def root_mean_sqaure_scale_error(a, b, c):
    # a: prediction, b: ground truth, c: reference lookback
    rmse = root_mean_square_error(a, b)
    naive_rmse = root_mean_square_error(c[:,1:], c[:,:-1])
    return rmse / naive_rmse





