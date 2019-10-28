import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from helper import getBatch_pred, getBatch_dis
from torch import optim
import numpy as np
from data import Sample
from replay import ReplayMemory
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

#Training examples for nll loss
def train_pred_each(generator, epoch, trainSample, optimizer, batch_size, embed_dim, recom_length, loss_fn_target, loss_fn_reward, device, generator_only = True, action_given = True, only_rewards = False):
    print('\nGENERATOR TRAINING : Epoch ' + str(epoch))
    generator.train()
    all_costs   = []
    logs        = []
    decay=0.95
    max_norm=5
    #loss_fn.size_average = False
    all_num=0
    last_time = time.time()
    correct = 0.
    correct_reward = 0.
    #mapeach=0.
    correctk = 0.
     
    #Adjust the learning rate 
    if epoch>1:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * decay
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))
     
    for stidx in range(0, trainSample.length(), batch_size):
        # prepare batch
        embed_batch, length, tgt_batch, reward_batch, action_batch = getBatch_pred(stidx, stidx + batch_size, trainSample, embed_dim, recom_length) 
        embed_batch,tgt_batch, reward_batch, action_batch = Variable(embed_batch.to(device)), Variable(tgt_batch.to(device)), Variable(reward_batch.to(device)), Variable(action_batch.to(device))
        k = embed_batch.size(0) #Actual batch size
        # model forward
        enc_out, h = generator((embed_batch, length))
        if generator_only:
            if action_given == True:
                output = generator.next_click(enc_out[:,-1,:], action_batch, len(embed_batch))                
            else:
                output = generator.next_simple(enc_out[:,-1,:])
        else:
            _, action, _ = agent((embed_batch, length))        
            output = generator.next_click(enc_out[:,-1,:], action, len(embed_batch))
        #Get next click
        reward, reward_logit = generator.get_reward(tgt_batch.view(-1,1), enc_out[:,-1,:].unsqueeze(0))
        all_prob_output = output.data.cpu().numpy()
        # reward correctness
        pred_reward = torch.round(reward.data)#.max(1)[1]
        correct_reward += pred_reward.long().eq(reward_batch.data.long()).cpu().sum().numpy()
        for i in range(len(all_prob_output)):
            pos = int(np.argwhere(np.argsort(-all_prob_output[i])==tgt_batch.data.long().cpu().numpy()[i])[0]+1)
            #mapeach += 1/pos
            # p@k
            if pos <= 1:
                correct += 1   
            if pos <= 10:
                correctk += 1
        # loss
        loss_pred = loss_fn_target(output, tgt_batch)
        #weight_loss = (reward_batch + 1) #** 5.3
        weight_loss = torch.FloatTensor(k).fill_(1).cuda() 
        loss_fn_reward = nn.BCEWithLogitsLoss(weight_loss)
        loss_fn_target.size_average = True
        loss_reward = loss_fn_reward(reward_logit, reward_batch)
        if not only_rewards:
            loss = loss_pred + loss_reward
        else:
            loss = loss_reward
            #Unable updates of the rnn model
            for name, param in generator.named_parameters():
                if 'embedding' in name or 'encoder' or 'enc2out' in name:
                    param.requires_grad = False
                    
        all_costs.append(loss.data.cpu().numpy())
        # backward
        optimizer.zero_grad()
        loss.backward()
        #Gradient clipping
        clip_grad_norm_(filter(lambda p: p.requires_grad, generator.parameters()), 5)
        #clip_grad_value_(filter(lambda p: p.requires_grad, generator.parameters()), 1)
        # optimizer step
        optimizer.step()
    train_acc = np.round(100 * correct/trainSample.length(), 2)
    #train_map=np.round(100 * mapeach/trainSample.length(), 2)
    train_preck=np.round(100 * correctk/trainSample.length(), 2)
    train_reward_acc = np.round(100 * correct_reward/trainSample.length(), 2)
    print('results : epoch {0} ; mean accuracy pred : {1}; mean P@10 pred: {2}; mean accuracy reward: {3}'.format(epoch, train_acc,train_preck, train_reward_acc))
    return train_acc, train_preck, np.mean(all_costs)
    
def train_dis_each(discriminator, epoch, trainSample, optimizer, batch_size, embed_dim, recom_length, loss_fn, device):
    print('\nDISCRIMINATOR TRAINING : Epoch ' + str(epoch))
    discriminator.train()
    all_costs   = []
    logs        = []
    decay=0.95
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
        embed_batch,length,tgt_batch, reward_batch, action_batch = getBatch_dis(stidx, stidx + batch_size, trainSample, embed_dim, recom_length)         
        reward_batch=reward_batch.type(torch.FloatTensor)
        embed_batch,tgt_batch, reward_batch, action_batch = Variable(embed_batch.to(device)), Variable(tgt_batch.to(device)), Variable(reward_batch.to(device)), Variable(action_batch.to(device))
        k = embed_batch.size(0) #Actual batch size
        # model forward
        output = discriminator((embed_batch, length), reward_batch, action_batch)
        
        all_prob = output.data.cpu().numpy()
        for i in range(len(all_prob)):
            pos = int((np.argwhere(np.argsort(-all_prob[i])==tgt_batch.data.long().cpu().numpy()[i])[0]+1))
            mapeach += 1/pos
            # p@k
            if pos <= 1:
                correct += 1      
        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data)
        # backward
        optimizer.zero_grad()
        loss.backward() 
        #Gradient clipping
        clip_grad_norm_(filter(lambda p: p.requires_grad, discriminator.parameters()), 5)
        
        # optimizer step
        optimizer.step()
    train_acc = np.round(100 * correct/trainSample.length(), 2)
    train_map=np.round(100 * mapeach/trainSample.length(), 2)
    #print('results : epoch {0} ; mean accuracy train : {1}; meanMAP: {2}'.format(epoch, train_acc,train_map))
    return train_acc, train_map

def train_gen_pg_each(generator, agent, discriminator, epoch, trainSample, subnum, optimizer_agent, optimizer_usr, batch_size, embed_dim, recom_length, max_length, real_label_num, device, gen_ratio, pretrain = False, shuffle_index=None):
    generator.train()
    agent.train()
    print('\nTRAINING : Epoch ' + str(epoch))
    generator.train()
    all_costs   = []
    logs        = []
    decay = 0.95
    gamma = 0.9
    max_norm=5
    all_num=0
    last_time = time.time()
     
    #Adjust the learning rate
    if epoch>1:
        optimizer_agent.param_groups[0]['lr'] = optimizer_agent.param_groups[0]['lr'] * decay
        optimizer_usr.param_groups[0]['lr'] = optimizer_usr.param_groups[0]['lr'] * decay
    print('Learning rate_agent : {0}'.format(optimizer_agent.param_groups[0]['lr']))
    print('Learning rate_usr : {0}'.format(optimizer_usr.param_groups[0]['lr']))
    
    #Generate subsamples
    trainSample_sub = Sample()
    trainSample_sub.subSample_copy(subnum, trainSample, shuffle_index)
    for stidx in range(0, trainSample_sub.length(), batch_size):
        # prepare batch
        embed_batch, length, _, reward_batch, action_batch = getBatch_dis(stidx, stidx + batch_size, trainSample_sub, embed_dim, recom_length) 
        embed_batch, reward_batch, action_batch = Variable(embed_batch.to(device)), Variable(reward_batch.to(device)), Variable(action_batch.to(device)) 
        k = embed_batch.size(0) #Actual batch size
        replay = ReplayMemory(generator, agent, int((1+gen_ratio)*k), max_length, real_label_num, action_batch.size(1))
        replay.init_click((embed_batch, length), reward_batch, action_batch)
        replay.gen_sample(batch_size, True, discriminator)
        tgt_reward, gen_reward, usr_prob, agent_prob = replay.tgt_rewards.type(torch.FloatTensor).to(device), replay.gen_rewards.type(torch.FloatTensor).to(device), replay.usr_probs.to(device), replay.agent_probs.to(device)
         
        tgt_prob = torch.abs(1.0-torch.round(tgt_reward)-tgt_reward)
        tgt_reward = torch.round(tgt_reward)
        if not pretrain: 
            loss_usr = -((torch.log(usr_prob + 1e-12) + torch.log(tgt_prob + 1e-12)) * gen_reward).sum()/k
        #Calculate the cumulative reward
        tgt_reward = gen_reward * (1 + tgt_reward)
        tgt_value = generator.value(tgt_reward)
        #loss_agent = -(torch.log(agent_prob + 1e-12) * (gen_reward + tgt_value)).sum()/k #+ 1e-18
        loss_agent = -(torch.log(agent_prob + 1e-12) * (tgt_value)).sum()/k #+ 1e-18
        all_costs.append(loss_agent.data.cpu().numpy())
        # backward
        optimizer_agent.zero_grad()
        optimizer_usr.zero_grad()
        if not pretrain:
            loss_usr.backward(retain_graph=True) 
            #Print gradients for each layer
            '''
            print("Gradients for user behavior models:")
            print("Embedding:")
            generator.embedding.print_grad()
            print("Encoder:")
            generator.encoder.print_grad()
            print("MLPlayer:")
            print(generator.enc2out.weight.grad)
            '''
            #Gradient clipping
            clip_grad_value_(filter(lambda p: p.requires_grad, generator.parameters()), 1)
            #clip_grad_norm_(filter(lambda p: p.requires_grad, generator.parameters()), 5)
            optimizer_usr.step()
        loss_agent.backward()
        #Gradient clipping
        clip_grad_value_(filter(lambda p: p.requires_grad, agent.parameters()), 1)
        #clip_grad_norm_(filter(lambda p: p.requires_grad, agent.parameters()), 5)
        # optimizer step
        optimizer_agent.step()
        # Printing
        if len(all_costs) == 100:
            logs.append( '{0} ; loss {1} ; seq/s {2}'.format(stidx, round(np.mean(all_costs),2), int(len(all_costs) * batch_size / (time.time() - last_time))))
            print(logs[-1])
            last_time = time.time()
            all_costs = []
    return all_costs
