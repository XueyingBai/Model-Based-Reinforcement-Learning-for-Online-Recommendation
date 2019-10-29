import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
from replay_simu import ReplayMemory

def train_gen_pg_each(policy, env, epoch, optimizer, num_clicks, recom_number, max_length, batch_size=256, total_size=10000):
    policy.train()
    env.eval()
    print('\nTRAINING : Epoch ' + str(epoch))
    all_costs   = []
    logs        = []
    decay=0.95
    max_norm=5
    all_num=0
    last_time = time.time()
    #Adjust the learning rate
    if epoch>1:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * decay
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))
    #Generate subsamples
    #for stidx in range(0, total_size, batch_size):
    for stidx in range(0, total_size, total_size):
        '''
        # prepare batch
        if stidx + batch_size < total_size:
            real_size = batch_size
        else:
            real_size = total_size - stidx
        '''
        real_size = total_size
        
        batch_replay = ReplayMemory(env, policy, real_size, max_length, num_clicks, recom_number)
        batch_replay.gen_sample(real_size)
        click_batch, reward_batch, action_batch, prob_batch = Variable(batch_replay.clicks), Variable(batch_replay.rewards), Variable(batch_replay.actions), Variable(batch_replay.probs, requires_grad = True) 
        value_batch = env.value(reward_batch)
        loss = -(torch.log(prob_batch) * (value_batch)).sum()
        all_costs.append(loss.data.cpu().numpy())
        # backward
        optimizer.zero_grad()
        loss.backward() 
        #Gradient clipping
        clip_grad_value_(filter(lambda p: p.requires_grad, policy.parameters()), 1)
        # optimizer step
        optimizer.step()
        #optimizer.param_groups[0]['lr'] = current_lr
        # Printing
        if len(all_costs) == 10000:
            logs.append( '{0} ; loss {1} ; seq/s {2}'.format(stidx, round(np.mean(all_costs),2), int(len(all_costs) * batch_size / (time.time() - last_time))))
            print(logs[-1])
            last_time = time.time()
            all_costs = []
    return all_costs, reward_batch.float().sum(1).mean().data.cpu().numpy()
