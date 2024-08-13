import torch
import time
import math
import numpy as np
from utils.utils_ import log_string, metric
from utils.utils_ import load_data

def test(args, log, device, model_fn, anchors, is_train=True):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, adj_mx, adj2, mean, std) = load_data(args)
    
    num_train, _, num_vertex = trainX.shape

    num_val = valX.shape[0]
    num_test = testX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)
    test_num_batch = math.ceil(num_test / args.batch_size)

    # test model
    log_string(log, '**** testing model ****')
    log_string(log, 'loading model from %s' % model_fn)
    model = torch.load(model_fn)
    log_string(log, 'model restored!')
    log_string(log, 'evaluating...')
    model.eval()

    with torch.no_grad():

        trainPred = []
        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            X = trainX[start_idx: end_idx].to(device)
            TE = trainTE[start_idx: end_idx].to(device)
            _,_,_,_,pred_batch = model(X, TE, is_train)
            trainPred.append(pred_batch.cpu().detach().clone())
            del X, TE, pred_batch
        trainPred = torch.from_numpy(np.concatenate(trainPred, axis=0))
        trainPred = trainPred * std + mean

        valPred = []
        for batch_idx in range(val_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
            X = valX[start_idx: end_idx].to(device)
            TE = valTE[start_idx: end_idx].to(device)
            _,_,_,_,pred_batch = model(X, TE, is_train)
            valPred.append(pred_batch.cpu().detach().clone())
            del X, TE, pred_batch
        valPred = torch.from_numpy(np.concatenate(valPred, axis=0))
        valPred = valPred * std + mean

        testPred = []
        start_test = time.time()
        for batch_idx in range(test_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
            X = testX[start_idx: end_idx].to(device)
            TE = testTE[start_idx: end_idx].to(device)
            _,_,_,_,pred_batch = model(X, TE, is_train)
            testPred.append(pred_batch.cpu().detach().clone())
            del X, TE, pred_batch
        testPred = torch.from_numpy(np.concatenate(testPred, axis=0))
        testPred = testPred* std + mean
    end_test = time.time()
    train_mae, train_rmse = metric(trainPred, trainY)
    val_rmse, val_mae = metric(valPred, valY)
    test_rmse, test_mae = metric(testPred, testY)

    # merge train, val and test Pred and Y
    trainp = trainPred.numpy()
    valp = valPred.numpy()
    testp = testPred.numpy()
    trainy = trainY.numpy()
    valy = valY.numpy()
    testy = testY.numpy()
    # concat
    Pred = np.concatenate((trainp, valp), axis=0)
    Y = np.concatenate((trainy, valy), axis=0)
    Pred = np.concatenate((Pred, testp), axis=0)
    Y = np.concatenate((Y, testy), axis=0)
    # saving into npz
    np.savez(args.model_file + '.npz', pred=Pred, y=Y)

    log_string(log, 'testing time: %.1fs' % (end_test - start_test))
    log_string(log, '                MAE\t\tRMSE')
    log_string(log, 'train            %.2f\t\t%.2f' %
               (train_mae, train_rmse))
    log_string(log, 'val              %.2f\t\t%.2f' %
               (val_mae, val_rmse))
    log_string(log, 'test             %.2f\t\t%.2f' %
               (test_mae, test_rmse))
    log_string(log, 'performance in each prediction step')
    MAE, RMSE, MAPE, R2 = [], [], [],[]
    for step in range(args.num_pred):
        rmse, mae = metric(testPred[:, step], testY[:, step])
        MAE.append(mae)
        RMSE.append(rmse)
        log_string(log, 'step: %02d  %.2f\t\t%.2f' %(step + 1, mae, rmse))
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    log_string(
        log, 'average:         %.2f\t\t%.2f' %
             (average_mae, average_rmse))

    return trainPred, valPred, testPred
