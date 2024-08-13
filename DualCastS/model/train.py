import time
import datetime
from utils.utils_ import log_string
from model.model_ import *
from utils.utils_ import load_data


def train(model, trainX, trainTE, trainY, valX, valTE, valY, mean, std, args, log, loss_criterion, kl_criterion,  optimizer, scheduler, device):
    
    num_train, _, num_vertex = trainX.shape
    
    log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)

    wait = 0
    val_loss_min = float('inf')
    train_total_loss = []
    val_total_loss = []

    # Train & validation
    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        # shuffle
        permutation = torch.randperm(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        # train
        start_train = time.time()
        model.train()
        train_loss = 0
        
        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            X = trainX[start_idx: end_idx].to(device)
            TE = trainTE[start_idx: end_idx].to(device)
            label = trainY[start_idx: end_idx].to(device)
            optimizer.zero_grad()
            gfeati, gfeate, p_Z, anchors, pred = model(X,TE)
            
            pred = pred * std + mean
            pred_loss = loss_criterion(pred, label)
            loss_batch = pred_loss
            
            if args.l_flt:
                l_flt = args.alpha/kl_criterion(F.log_softmax(gfeati,dim=-1), F.softmax(gfeate,dim=-1))
                loss_batch += l_flt

            if args.l_env:          
                gfeate_permute = gfeate[torch.randperm(gfeate.size(0))]
                l_env = args.beta / kl_criterion(F.log_softmax(gfeate_permute,dim=-1), F.softmax(gfeate,dim=-1))
                loss_batch += l_env
            
            if args.l_dbi:
                anchors = anchors.view(args.num_prototypes, -1)
            
                s = []
                for i in range(args.num_prototypes):
                    n = len(p_Z[i])
                    if n > 1:
                        # updating the anchors
                        tmp_p = torch.stack(p_Z[i], dim=0)
                        tmp_p = tmp_p.reshape(n, -1)
                        tmp_loss = torch.norm(anchors[i]-tmp_p, p=2)/n
                    elif n == 1:
                        tmp_p = p_Z[i][0]
                        tmp_p = tmp_p.reshape(n, -1)
                        tmp_loss = torch.norm(anchors[i]-tmp_p, p=2)/n
                    else:
                        tmp_loss = torch.tensor(0.0).to(gfeati.device)
                    s.append(tmp_loss)

                # distance among vectors
                dbi = torch.tensor(0.0).to(gfeati.device)
                num_valid_types = 0
                for i in range(args.num_prototypes):
                    if s[i] == 0:
                        continue
                    else:
                        r = []
                        for j in range(args.num_prototypes):
                            if i != j and s[j] != 0:
                                dist = torch.norm(anchors[i]-anchors[j], p=2)
                            # print(dist)
                                r.append((s[i]+s[j])/dist)
                        if len(r) != 0:
                            num_valid_types += 1
                            dbi += torch.max(torch.tensor(r))
                    
                # intra_loss = intra_loss/args.prototypes
                dbi = dbi/num_valid_types

                l_dbi = args.gamma * dbi
                loss_batch += l_dbi

                
            train_loss += float(loss_batch) * (end_idx - start_idx)
            loss_batch.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (batch_idx+1) % 10 == 0:
                if args.l_flt and args.l_dbi and args.l_env:
                    print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}, pred_loss:{pred_loss:.4f}, l_flt:{l_flt:.4f}, l_env:{l_env:.4f}, l_dbi:{l_dbi:.4f}')
                elif args.l_flt and args.l_env:
                    print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}, pred_loss:{pred_loss:.4f}, l_flt:{l_flt:.4f}, l_env:{l_env:.4f}')
                elif args.l_dbi and args.l_env:
                    print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}, pred_loss:{pred_loss:.4f}, l_dbi:{l_dbi:.4f}, l_env:{l_env:.4f}')
                elif args.l_flt and args.l_dbi:
                    print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}, pred_loss:{pred_loss:.4f}, l_flt:{l_flt:.4f}, l_dbi:{l_dbi:.4f}')
                elif args.l_flt:
                    print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}, pred_loss:{pred_loss:.4f}, l_flt:{l_flt:.4f}')
                elif args.l_dbi:
                    print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}, pred_loss:{pred_loss:.4f}, l_dbi:{l_dbi:.4f}')
                elif args.l_env:
                    print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}, pred_loss:{pred_loss:.4f}, l_env:{l_env:.4f}')
                else:
                    print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}, pred_loss:{pred_loss:.4f}')
            del X, TE, label, pred, loss_batch
        train_loss /= num_train
        train_total_loss.append(train_loss)
        end_train = time.time()

        # val loss
        start_val = time.time()
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx in range(val_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                X = valX[start_idx: end_idx].to(device)
                TE = valTE[start_idx: end_idx].to(device)
                label = valY[start_idx: end_idx].to(device)
                _,_,_,_,pred = model(X,TE)
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, label)
                val_loss += loss_batch * (end_idx - start_idx)
                del X, TE, label, pred, loss_batch
        val_loss /= num_val
        val_total_loss.append(val_loss)
        end_val = time.time()
        log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             args.max_epoch, end_train - start_train, end_val - start_val))
        log_string(
            log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
        if val_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.model_file}.pkl')
            wait = 0
            val_loss_min = val_loss
            torch.save(model, args.model_file+".pkl")
        else:
            wait += 1
        scheduler.step()

    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}.pkl')
    return train_total_loss, val_total_loss
