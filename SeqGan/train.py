import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from helper import getBatch, roll_out
from torch import optim
import numpy as np
from data import Sample
#Training examples for nll loss(pretraining)
def train_nll(model_type, model, epoch, trainSample, optimizer, batch_size, embed_dim, loss_fn, device):
    print('\nTRAINING : Epoch ' + str(epoch))
    model.train()
    all_costs   = []
    logs        = []
    decay=0.99
    max_norm=5
    #loss_fn.size_average = False
    all_num=0
    last_time = time.time()
    correct = 0.
    mapeach=0.
    #Adjust the learning rate
    if epoch>1:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * decay
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, trainSample.length(), batch_size):
        # prepare batch
        embed_batch,length,tgt_batch, pur_batch = getBatch(stidx, stidx + batch_size, trainSample, embed_dim) 
        embed_batch,tgt_batch = Variable(embed_batch.to(device)), Variable(tgt_batch.to(device))
        k = embed_batch.size(0) #Actual batch size
        # model forward
        if model_type == 'generator':
            output, _ = model((embed_batch, length))
        else:
            output = model((embed_batch, length))
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().numpy()
        all_prob = output.data.cpu().numpy()
        for i in range(len(all_prob)):
            mapeach += 1/int((np.argwhere(np.argsort(-all_prob[i])==tgt_batch.data.long().cpu().numpy()[i])[0]+1))
        assert len(pred) == k        
        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data)
        # backward
        optimizer.zero_grad()
        loss.backward() 
        
        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0        
        for p in model.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)        
        if total_norm > max_norm:
            shrink_factor = max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update
        
        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr
        # Printing
        if len(all_costs) == 500:
            logs.append( '{0} ; loss {1} ; seq/s {2} ; accuracy train : {3}; map train : {4}'.format(
                    stidx, round(np.mean(all_costs),2), int(len(all_costs) * batch_size / (time.time() - last_time)),
                    round(100.*correct/(stidx+k), 2), round(100.*mapeach/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            all_costs = []
    train_acc = np.round(100 * correct/trainSample.length(), 2)
    train_map=np.round(100 * mapeach/trainSample.length(), 2)
    print('results : epoch {0} ; mean accuracy train : {1}; meanMAP: {2}'.format(epoch, train_acc,train_map))
    return train_acc, train_map

def train_gen_pg(generator, discriminator, epoch, trainSample, subnum, optimizer, batch_size, embed_dim, max_length, sample_num, device):
    generator.train()
    print('\nTRAINING : Epoch ' + str(epoch))
    generator.train()
    all_costs   = []
    logs        = []
    decay=0.99
    max_norm=5
    all_num=0
    last_time = time.time()
    #Adjust the learning rate
    if epoch>1:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * decay
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))
    #Generate subsamples
    trainSample_sub = Sample()
    trainSample_sub.subSample_copy(subnum, trainSample)
    for stidx in range(0, trainSample_sub.length(), batch_size):
        # prepare batch
        embed_batch,length,tgt_batch, pur_batch = getBatch(stidx, stidx + batch_size, trainSample_sub, embed_dim) 
        embed_batch,pur_batch = Variable(embed_batch.to(device)), Variable(pur_batch.to(device)), 
        k = embed_batch.size(0) #Actual batch size
        gen_seq, gen_length, prob, _ = generator.sample((embed_batch, length), max_length, pur_batch)
        gen_seq, prob = gen_seq.to(device), prob.to(device)
        reward = roll_out(generator, discriminator, (gen_seq, gen_length), length, max_length, pur_batch, sample_num, device)         # loss
        loss = -(prob * reward).sum()# - reward_pur.sum()
        all_costs.append(loss.data)
        # backward
        optimizer.zero_grad()
        loss.backward() 
        
        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0        
        for p in generator.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)        
        if total_norm > max_norm:
            shrink_factor = max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update
        
        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr
        # Printing
        if len(all_costs) == 10:
            logs.append( '{0} ; loss {1} ; seq/s {2}'.format(stidx, round(np.mean(all_costs),2), int(len(all_costs) * batch_size / (time.time() - last_time))))
            print(logs[-1])
            last_time = time.time()
            all_costs = []
    return all_costs