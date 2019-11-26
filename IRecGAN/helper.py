import numpy as np
import torch
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from replay import ReplayMemory
#from torch.autograd import Variable

#The reward is a single value    
def getBatch_pred(batchstart, batchend, samples, embed_dim, recom_length):
    batchItem, batchTarget, batchReward, batchAction=samples.batchSample(batchstart, batchend)
    lengths = np.array([len(x) for x in batchItem])
    max_len = int(np.max(lengths))
    #batch first
    item=torch.LongTensor(len(batchItem), max_len).zero_()
    action = torch.LongTensor(len(batchItem), recom_length).zero_()
    for i in range(len(batchItem)):
        #print(torch.from_numpy(np.array(batchAction[i])))
        action[i, :len(batchAction[i])] = torch.from_numpy(np.array(batchAction[i]))
        action[i, len(batchAction[i]):] = int(0)
        #Batch first
        for j in range(len(batchItem[i])):
            item[i,j] = int(batchItem[i][j])
    return item, lengths, torch.LongTensor(np.array(batchTarget)), torch.FloatTensor(np.array(batchReward)), action

#The reward is a sequence    
def getBatch_dis(batchstart, batchend, samples, embed_dim, recom_length):
    batchItem, batchTarget, batchReward, batchAction=samples.batchSample(batchstart, batchend)
    lengths = np.array([len(x) for x in batchItem])
    max_len = int(np.max(lengths))
    #batch first
    item=torch.LongTensor(len(batchItem), max_len).zero_()
    reward=torch.LongTensor(len(batchItem), max_len).zero_()
    action=torch.LongTensor(len(batchItem), recom_length, max_len).zero_()
    for i in range(len(batchItem)):
        #Batch first
        for j in range(len(batchItem[i])):
            item[i,j] = int(batchItem[i][j])
            reward[i,j] = int(batchReward[i][j])
            action[i,:len(batchAction[i][j]),j] = torch.from_numpy(np.array(batchAction[i][j]))
            action[i,len(batchAction[i][j]):,j] = int(0)
    return item, lengths, torch.LongTensor(np.array(batchTarget)), reward, action

def split_index(train_ratio, dev_ratio, total_length, shuffle = False):
    #index=np.random.permutation(total_length)
    #Data shuffled by preprocessing during the generation
    if shuffle == True:
        index=np.random.permutation(total_length)
    else:
        index = np.array(range(total_length))
    #Get split
    trainnum=int(total_length*train_ratio)
    trainindex=index[:trainnum]
    validnum=int(total_length*dev_ratio)
    validindex=index[trainnum:trainnum+validnum]
    testindex=index[trainnum+validnum:total_length]
    return trainindex, validindex, testindex
    
def write_seq(seqlist, filename1, filename2, write_type='dis', real=True):
    click=open(os.path.join('',filename1),'a+') 
    if write_type == 'dis':
        target=open(os.path.join('',filename2),'a+') 
    for seq in seqlist:
        for item in seq:
            click.write(str(item) + ' ')
        click.write('\n')
        #If 'dis', then used for the discriminator training, need target
        if write_type == 'dis':
            if real == True:
                target.write('1\n')
            else:
                target.write('0\n')
    click.close()
    if write_type == 'dis':
        target.close()
    
def write_seq_reward(rewardlist, filename):
    reward=open(os.path.join('',filename),'a+') 
    for rew in rewardlist:
        for item in rew:
            reward.write(format(item, '.0f') + ' ')
        reward.write('\n')
    reward.close()
    
def write_seq_action(actionlist, filename):
    action=open(os.path.join('',filename),'a+')
    for act in actionlist:
        for item in act:
            for i in range(len(item)-1):
                action.write(str(item[i])+',')
            action.write(str(item[-1])+' ')
        action.write('\n')
    action.close()
    
def write_tensor(seq_samples, lengths, filename1, filename2, write_type='dis', real=False):
    click=open(os.path.join('',filename1),'a+') 
    target=open(os.path.join('',filename2),'a+') 
    seq_samples = seq_samples.data.cpu().numpy()
    for i in range(len(lengths)):
        for j in range(lengths[i]):
            click.write(str(seq_samples[i,j]) + ' ')
        click.write('\n')
        #If 'dis', then used for the discriminator training, need target
        if write_type == 'dis':
            if real == True:
                target.write('1\n')
            else:
                target.write('0\n')
    click.close()
    target.close()
    
def write_tensor_reward(seq_samples, lengths, filename):
    reward=open(os.path.join('',filename),'a+') 
    seq_samples = seq_samples.data.cpu().numpy()
    for i in range(len(lengths)):
        for j in range(lengths[i]):
            reward.write(str(seq_samples[i,j]) + ' ')
        reward.write('\n')
    reward.close()
    
def write_tensor_action(seq_samples, lengths, filename):
    action=open(os.path.join('',filename),'a+') 
    seq_samples = seq_samples.data.cpu().numpy()
    for i in range(len(lengths)):
        for j in range(lengths[i]):
            for k in range(seq_samples.shape[1]-1):
                action.write(str(seq_samples[i,k,j]) + ',')
            action.write(str(seq_samples[i,seq_samples.shape[1]-1,j]))
            if j != lengths[i]-1:
                action.write(' ')
        action.write('\n')
    action.close()
                    
def gen_fake(generator, agent, trainSample, batch_size, embed_dim, device, write_item, write_target, write_reward, write_action, action_num, max_length=5, recom_length=None):
    for stidx in range(0, trainSample.length(), batch_size):
        # prepare batch
        click_batch, length, _, reward_batch, action_batch = getBatch_dis(stidx, stidx + batch_size, trainSample, embed_dim, recom_length) 
        click_batch = click_batch.to(device)
        reward_batch = reward_batch.to(device)
        action_batch = action_batch.to(device)
        if recom_length == None:
            recom_length = action_batch.size(1)
        replay = ReplayMemory(generator, agent, len(length), max_length, action_num, recom_length)
        with torch.no_grad():
            replay.init_click_sample((click_batch, length), reward_batch, action_batch)
            replay.gen_sample(batch_size, False)
            seq_samples, lengths, seq_rewards, seq_actions = replay.clicks, replay.lengths, replay.tgt_rewards, replay.actions
            seq_rewards = torch.round(seq_rewards)
        write_tensor(seq_samples, lengths, write_item, write_target, 'dis', real=False)
        write_tensor_reward(seq_rewards, lengths, write_reward)
        write_tensor_action(seq_actions, lengths, write_action)
    return seq_samples, lengths, seq_rewards, seq_actions
        
def plot_data_dist(itemlist, filepath):
    item_dist = np.zeros(len(itemlist))
    for key, value in itemlist.items():
        item_dist[int(key)]=value
    fig, ax = plt.subplots()
    y_pos = np.arange(len(itemlist))
    ax.bar(y_pos, np.log(item_dist), align='center', alpha=0.5)
    fig.savefig(filepath)
    plt.close(fig)
        
def save_plot(epoch_num, step, value_losses, filepath, step0 = True):
    """Generate and save the plot"""
    fig, ax = plt.subplots()
    if step0:
        ax.plot(range(0, epoch_num + 1, step), value_losses)#,'.')
        ax.plot(range(0, epoch_num + 1, step), np.ones(len(range(0, epoch_num + 1, step)))*value_losses[0], 'r')
    else:
        ax.plot(range(0, epoch_num, step), value_losses)#,'.')
        ax.plot(range(0, epoch_num, step), np.ones(len(range(0, epoch_num, step)))*value_losses[0], 'r')
    fig.savefig(filepath)
    plt.close(fig)  # close the figure       