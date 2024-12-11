import os
import random
import numpy as np
import argparse
import time
from tqdm import tqdm
import shutil
import pandas as pd

import torch
import torch.nn as nn
from torch import optim

from util.data import batch_sampled_data, batch_data, to_one_hot, TimeSeriesDataset, TimeSeriesDataLoader
from util.tool import adjust_learning_rate, EarlyStopping
from util.metric import quantile_loss_tensor
from data_formatters.electricity import ElectricityFormatter
# from model.rnn import RNN
# from model.nbeats import NBeats
# from model.tft import TemporalFusionTransformer
# from model.mqcnn import MQCNN
# from model.mqt import MQT
from model.tat import TAT
# from model.patchtst import PatchTST
# from model.itransformer import iTransformer
# from model.tsmixer import TSMixer

import pdb

root = '.'

def train(model, configs):
    device = configs.device

    # np.random.seed(42)
    # data_formatter = ElectricityFormatter()
    # raw_data = pd.read_csv('./dataset/data/electricity/hourly_electricity.csv', index_col=0)
    # _, _, _ = data_formatter.split_data(raw_data)

    x_train_inputs = np.load('./dataset/data/electricity/process_train_inputs.npy')
    x_train_static = np.load('./dataset/data/electricity/process_train_identifier.npy', allow_pickle=True)

    x_valid_inputs = np.load('./dataset/data/electricity/process_valid_inputs.npy')
    x_valid_static = np.load('./dataset/data/electricity/process_valid_identifier.npy', allow_pickle=True)

    x_test_inputs = np.load('./dataset/data/electricity/process_test_inputs.npy')
    x_test_static = np.load('./dataset/data/electricity/process_test_identifier.npy', allow_pickle=True)

    print("Finish loading data")

    unique_categories = np.unique(x_train_static[:,0].flatten())
    cat_map = {category: idx for idx, category in enumerate(unique_categories)}

    x_train_static = to_one_hot(x_train_static, cat_map)
    x_valid_static = to_one_hot(x_valid_static, cat_map)
    x_test_static = to_one_hot(x_test_static, cat_map)

    xt_train, xt_valid, xt_test = torch.from_numpy(x_train_inputs).float(), torch.from_numpy(x_valid_inputs).float(), torch.from_numpy(x_test_inputs).float()
    xs_train, xs_valid, xs_test = torch.from_numpy(x_train_static).float(), torch.from_numpy(x_valid_static).float(), torch.from_numpy(x_test_static).float()

    print("Finish processing data")

    train_set = TimeSeriesDataset(xt_train, xs_train, horizon=configs.horizon, lookback=configs.lookback)
    vali_set  = TimeSeriesDataset(xt_valid, xs_valid, horizon=configs.horizon, lookback=configs.lookback)
    test_set  = TimeSeriesDataset(xt_test, xs_test, horizon=configs.horizon, lookback=configs.lookback)

    train_loader = TimeSeriesDataLoader(train_set, batchsize=configs.batchsize, shuffle=True)
    vali_loader  = TimeSeriesDataLoader(vali_set, batchsize=configs.batchsize, shuffle=False)
    test_loader  = TimeSeriesDataLoader(test_set, batchsize=configs.batchsize, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    # criterion = nn.MSELoss()
    criterion_50 = quantile_loss_tensor(0.5)
    criterion_90 = quantile_loss_tensor(0.9)
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True)

    print("Training start")

    for epoch in range(configs.epoch):
        iter_count = 0
        train_loss = []

        model.to(device)
        model.train()

        for i, (batch_xt, batch_xf, batch_xs, batch_y) in tqdm(enumerate(train_loader)):
            iter_count += 1
            optimizer.zero_grad()

            batch_xt = batch_xt.to(device)
            batch_xf = batch_xf.to(device)
            batch_xs = batch_xs.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_xt, batch_xf, batch_xs)
            outputs_50 = outputs[:,-configs.horizon:,0]
            outputs_90 = outputs[:,-configs.horizon:,1]
            loss = criterion_50(outputs_50, batch_y[:, -configs.horizon:, -1]) + criterion_90(outputs_90, batch_y[:, -configs.horizon:, -1])
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        # pdb.set_trace()

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_loader, criterion_50, criterion_90, device, configs)
        test_loss = test(model, test_loader, criterion_50, criterion_90, device, configs, flag=False)

        print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))
        early_stopping(vali_loss, model, path=root + '/electricity_checkpoint/' + configs.model_name + '_checkpoint_{}.pth'.format(configs.seed))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch + 1, configs)

    best_model_path =  root + '/electricity_checkpoint/' + configs.model_name + '_checkpoint_{}.pth'.format(configs.seed)
    model.load_state_dict(torch.load(best_model_path))
    test_loss = test(model, test_loader, criterion_50, criterion_90, device, configs, flag=True)



def vali(model, valid_loader, criterion_50, criterion_90, device, configs):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_xt, batch_xf, batch_xs, batch_y) in enumerate(valid_loader):

            batch_xt = batch_xt.to(device)
            batch_xf = batch_xf.to(device)
            batch_xs = batch_xs.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_xt, batch_xf, batch_xs)
            outputs_50 = outputs[:,-configs.horizon:,0]
            outputs_90 = outputs[:,-configs.horizon:,1]
            loss = criterion_50(outputs_50, batch_y[:, -configs.horizon:, -1]) + criterion_90(outputs_90, batch_y[:, -configs.horizon:, -1])
            total_loss.append(loss.item())

    # pdb.set_trace()
    total_loss = np.average(total_loss)
    return total_loss


def test(model, test_loader, criterion_50, criterion_90, device, configs, flag=False):
    preds = []
    trues = []

    total_loss = []
    model.eval()
    with torch.no_grad():  
        for i, (batch_xt, batch_xf, batch_xs, batch_y) in enumerate(test_loader):
            batch_xt = batch_xt.float().to(device)
            batch_xf = batch_xf.float().to(device)
            batch_xs = batch_xs.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_xt, batch_xf, batch_xs)

            pred = outputs[:, -configs.horizon:, :].detach().cpu()
            pred_50 = outputs[:, -configs.horizon:, 0].detach().cpu()
            pred_90 = outputs[:, -configs.horizon:, 1].detach().cpu()
            true = batch_y[:, -configs.horizon:, -1].detach().cpu()

            loss = criterion_50(pred_50, true[:, -configs.horizon:]) + criterion_90(pred_90, true[:, -configs.horizon:])

            total_loss.append(loss.item())
            preds.append(pred.numpy())
            trues.append(true.numpy())

    total_loss = np.average(total_loss)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    print('test shape:', preds.shape, trues.shape)

    print('test mse:{}'.format(total_loss))
    if flag:
        np.save(root + '/electricity_results/' + configs.model_name + '_pred_{}.npy'.format(configs.seed), preds)

    return total_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='electricity')

    # exp/data setup
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--lookback', type=int, default=168, help='lookback window size, past 6 months')
    parser.add_argument('--horizon', type=int, default=24, help='horizon window size, next 1 month')
    # parser.add_argument('--num_class', type=int, default=371, help='number of static classes')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu/cpu device')
    parser.add_argument('--num_xt', type=int, default=1, help='number of target time series')
    parser.add_argument('--num_xf', type=int, default=4, help='number of future time series')
    parser.add_argument('--num_xs', type=int, default=369, help='number of static classes')

    # train hyperparameter
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=512, help='learning rate')
    parser.add_argument('--epoch', type=int, default=20, help='training epochs')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate: 1=exp decay, 2=multistep, 3=no decay')

    parser.add_argument('--use_xf', type=bool, default=True, help='use future features')
    parser.add_argument('--use_xs', type=bool, default=True, help='use static features')

    # model hyperparameter
    parser.add_argument('--model_name', type=str, default='tsmixer', help='model selection')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--hidden', type=int, default=160, help='hidden size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    # nbeats hyperparameter
    parser.add_argument('--degree_of_polynomial', type=int, default=3, help='degree of polynomial in trend block')
    parser.add_argument('--num_of_harmonics', type=int, default=1, help='number of harmonics in seasonality block')
    parser.add_argument('--num_block', type=int, default=4, help='number of nbeats blocks')

    # tft hyperparameter
    parser.add_argument('--heads', type=int, default=4, help='number of heads in multi-head attention')

    # mqcnn hyperparameter
    parser.add_argument('--atrous_rates', type=list, default=[1, 2, 4, 8, 16, 32], help='atrous rates')

    # mqt hyperparameter
    parser.add_argument('--grain', type=int, default=7, help='grain')
    parser.add_argument('--units_per_horizon', type=int, default=10, help='decoder hidden size per horizon')
    parser.add_argument('--position_global_atrous_rates', type=list, default=[1, 2, 4, 8, 8], help='positional encoding atrous rate')


    args = parser.parse_args()

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    
    model_dict = {
        # 'rnn'   : RNN,
        # 'deepar': None,
        # 'nbeats': NBeats,
        # 'mqcnn' : MQCNN,
        # 'mqt'   : MQT,
        # 'tft'   : TemporalFusionTransformer,
        # 'patchtst'   : PatchTST,
        # 'itrans': iTransformer,
        'tat'   : TAT,
        # 'tsmixer'    : TSMixer
    }

    model = model_dict[args.model_name](args)

    # if os.path.exists('./checkpoint'):
    #     shutil.rmtree('./checkpoint')
    if not os.path.exists(root + '/electricity_checkpoint'):
        os.mkdir(root + '/electricity_checkpoint')

    if not os.path.exists(root + '/electricity_results'):
        os.mkdir(root + '/electricity_results')

    train(model, args)