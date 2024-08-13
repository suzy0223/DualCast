import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def metric(pred,truth):
    pred = pred.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()
    idx = truth > 0
    RMSE = np.sqrt(np.mean((pred[idx] - truth[idx]) ** 2))
    MAE = np.mean(np.abs(pred[idx] - truth[idx]))
    
    return RMSE, MAE


def gen_high_order_adj(adj_mx, order):

    adjout = np.linalg.matrix_power(adj_mx, order) - np.linalg.matrix_power(adj_mx, order-1)

    return adjout

def seq2instance(data, num_his, num_pred):
    if len(data.shape) == 2:
        num_step, dims = data.shape
        num_sample = num_step - num_his - num_pred + 1
        x = torch.zeros(num_sample, num_his, dims)
        y = torch.zeros(num_sample, num_pred, dims)
    else:
        num_step, N, dims = data.shape
        num_sample = num_step - num_his - num_pred + 1
        x = torch.zeros(num_sample, num_his, N, dims)
        y = torch.zeros(num_sample, num_pred, N, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y


def load_adj_fn(args,node_num):
    # load adjacency matrix from file
    adj_mx = np.zeros((node_num,node_num))
    with open(args.Adj_file, 'r') as f:
        # read line until end of file
        line = f.readlines()
    
    for i in range(len(line)):
        line[i] = line[i].strip('\n')
        line[i] = line[i].split(' ')
        adj_mx[int(line[i][0]),int(line[i][1])] = float(line[i][2])

    return adj_mx


def load_data(args):
    if args.traffic_file == './data/PEMS08.npz':
        if args.traffic_type == 'speed':
            traffic = np.load(args.traffic_file)['data'][:,:,-1]
        elif args.traffic_type == 'flow':
            traffic = np.load(args.traffic_file)['data'][:,:,0]
        num_step,num_node = traffic.shape
        dayofweek = list(set(range(0, num_step)))
        dayofweek = np.array(dayofweek)
        dayofweek = (dayofweek / args.T)%7
        dayofweek = dayofweek.astype(np.int32)
        dayofweek = np.reshape(dayofweek, newshape = (-1, 1))
        dayofweek = torch.tensor(dayofweek)

        is_holiday = list(set(range(0, num_step)))
        is_holiday = np.array(is_holiday)
        is_holiday = (is_holiday / args.T) % 7
        is_holiday = is_holiday.astype(np.int32)
        is_holiday[is_holiday >= 3] = 0
        is_holiday[is_holiday == 2] = 1
        is_holiday[3*args.T:4*args.T] = 1
        is_holiday = np.reshape(is_holiday, newshape = (-1, 1))
        is_holiday = torch.tensor(is_holiday)
    elif args.traffic_file == './data/PEMSD3.npy':
        # 20180901 (Sat) - 20181130 91 days
        traffic = np.load(args.traffic_file)
        num_step,num_node = traffic.shape
        
        dayofweek = list(set(range(0, num_step)))
        dayofweek = np.array(dayofweek)
        dayofweek = ((dayofweek / args.T)+1)%7
        dayofweek = dayofweek.astype(np.int32)
        dayofweek = np.reshape(dayofweek, newshape = (-1, 1))
        dayofweek = torch.tensor(dayofweek)

        is_holiday = list(set(range(0, num_step)))
        is_holiday = np.array(is_holiday)
        is_holiday = ((is_holiday / args.T)+1) % 7
        is_holiday = is_holiday.astype(np.int32)
        is_holiday[is_holiday >= 3] = 0
        is_holiday[is_holiday == 2] = 1
        # 9.3; 10.8;11.11;11.22; holiday
        is_holiday[2*args.T:3*args.T] = 1
        is_holiday[37*args.T:38*args.T] = 1
        is_holiday[71*args.T:72*args.T] = 1
        is_holiday[82*args.T:83*args.T] = 1
        is_holiday = np.reshape(is_holiday, newshape = (-1, 1))
        is_holiday = torch.tensor(is_holiday)

    traffic = torch.from_numpy(traffic)
    print("num_step = {}, num_node = {}".format(num_step, num_node))

    timeofday = list(set(range(0, num_step)))
    timeofday = np.array(timeofday)
    timeofday = timeofday % args.T
    timeofday = timeofday.astype(np.int32)
    timeofday = np.reshape(timeofday, newshape = (-1, 1))
    timeofday = torch.tensor(timeofday)

    time = torch.cat((dayofweek, timeofday, is_holiday), -1)
    
    print("Time shape = {}".format(time.shape))

    adj_mx = load_adj_fn(args,num_node)
    # set adj_mx to 0,1
    adj_mx = np.where(adj_mx > 0, 1, 0)

    adj2 = gen_high_order_adj(adj_mx, 2)

    adj_mx = adj_mx-np.eye(num_node)
    adj2 = torch.from_numpy(adj2)
    adj2.to(torch.float32)
    adj_mx = adj_mx+np.eye(num_node)
    adj_mx = torch.from_numpy(adj_mx)
    adj_mx.to(torch.float32)

    print("Time shape = {}".format(time.shape))
    trafficx,trafficy = seq2instance(traffic, args.num_his, args.num_pred)
    time = seq2instance(time, args.num_his, args.num_pred)
    time = torch.cat(time, 1).type(torch.int32)
    print("Time shape = {}".format(time.shape))

    samples = trafficx.shape[0]
    train_steps = round(args.train_ratio * samples)
    test_steps = round(samples*args.test_ratio)
    val_steps = samples - train_steps - test_steps
    trainX = trafficx[: train_steps]
    trainY = trafficy[: train_steps]
    valX = trafficx[train_steps: train_steps + val_steps]
    valY = trafficy[train_steps: train_steps + val_steps]
    testX = trafficx[-test_steps:]
    testY = trafficy[-test_steps:]
    
    # normalization
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # train/val/test
    trainTE = time[: train_steps]
    valTE = time[train_steps: train_steps + val_steps]
    testTE = time[-test_steps:]

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, adj_mx, adj2, mean, std)


# dataset creation
class dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.len = data_x.shape[0]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# plot test results
def save_test_result(trainPred, trainY, valPred, valY, testPred, testY):
    with open('./figure/test_results.txt', 'w+') as f:
        for l in (trainPred, trainY, valPred, valY, testPred, testY):
            f.write(list(l))
