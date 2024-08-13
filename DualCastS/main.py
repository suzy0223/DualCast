
import argparse
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np

from utils.utils_ import log_string
from utils.utils_ import count_parameters, load_data, str2bool

from model.model_ import STTN
from model.train import train
from model.test import test
import torch
import os

torch.cuda.empty_cache()
"""adding seed for reproducibility"""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=12,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=12,
                    help='prediction steps')
parser.add_argument('--input_dim', type=int, default=1,
                    help='history steps')
parser.add_argument('--output_dim', type=int, default=1,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--embed_dim', type=int, default=64,
                    help='embed_dim')
parser.add_argument('--forward_expansion', type=int, default=4,
                    help='forward_expansion')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=100,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--dropout_rate', type=float, default=0,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='decay epoch')
parser.add_argument('--traffic_file', default='./data/PEMS08.npz',
                    help='traffic file')
parser.add_argument('--Adj_file', default='./data/Adj08.txt',
                    help='adjacent matrix file base on road netwok')
parser.add_argument('--model_file', default='./data/STTN_PEMS08_all_L',
                    help='save the model to disk')
parser.add_argument('--log_file', default='./data/log_PEMS08_all_L',
                    help='log file')
parser.add_argument('--T', type=int, default=288,
                    help='time slot num per day')
parser.add_argument('--traffic_type', type=str, default="flow",
                    help='traffic condition type')
parser.add_argument('--run_count', type=int, default=1,
                    help='times to run this params setting')
parser.add_argument('--num_prototypes', type=int, default=17,
                    help='number of prototypes')
parser.add_argument('--l_flt', type=str2bool, default=True,
                    help='if use flt loss')
parser.add_argument('--l_dbi', type=str2bool, default=True,
                    help='if use dbi loss')
parser.add_argument('--l_env', type=str2bool, default=True,
                    help='if use dbi loss')
parser.add_argument('--alpha', type=float, default=1,
                    help='coefficient of filter loss')
parser.add_argument('--beta', type=float, default=0.01,
                    help='coefficient of environment loss')
parser.add_argument('--gamma', type=float, default=0.05,
                    help='coefficient of dbi loss')

args = parser.parse_args()
args.log_file = args.log_file +"_"+ str(args.L) +"_K"+ str(args.K) +"_alpha"+ str(args.alpha) + "_beta" + str(args.beta)+ "_gamma" + str(args.gamma)+"_"+ str(args.run_count)
args.model_file = args.model_file +"_"+ str(args.L)+"_"+ str(args.K) +"__alpha"+ str(args.alpha) + "_beta" + str(args.beta)+ "_gamma" + str(args.gamma)+"_"+ str(args.run_count)
log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])
T = 24 * 60 // args.time_slot  # Number of time steps in one day
# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, adj_mx, adj2, mean, std) = load_data(args)
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
log_string(log, 'data loaded!')

# build model
log_string(log, 'compiling model...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = STTN(args, adj_mx.to(torch.float).to(device),device)
model.to(device)
loss_criterion = nn.L1Loss()
kl_criterion = nn.KLDivLoss(reduction='batchmean')

optimizer = optim.Adam(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.decay_epoch,
                                      gamma=0.9)
parameters = count_parameters(model)
log_string(log, 'trainable parameters: {:,}'.format(parameters))

if __name__ == '__main__':
    start = time.time()
    loss_train, loss_val = train(model, trainX, trainTE, trainY, valX, valTE, valY, mean, std, args, log, loss_criterion, kl_criterion, optimizer, scheduler, device)
    trainPred, valPred, testPred = test(args, log, device)
    end = time.time()
    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()
    