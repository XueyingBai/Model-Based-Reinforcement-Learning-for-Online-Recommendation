import numpy as np
import torch
import os
from generator import Generator
#from torch.autograd import Variable
def getBatch(batchstart, batchend, samples, embed_dim):
    batchItem, batchTarget, batchPurchase=samples.batchSample(batchstart, batchend)
    #print(batchPurchase)
    lengths = np.array([len(x) for x in batchItem])
    lengths_pur = np.array([len(x) for x in batchPurchase])
    max_len = int(np.max(lengths))
    max_pur = int(np.max(lengths_pur))
    #batch first
    item=torch.LongTensor(len(batchItem), max_len).zero_()
    purchase=-torch.ones(len(batchPurchase), max_pur, dtype=torch.long) #initialize the tensor with -1 
    for i in range(len(batchItem)):
        #Batch first
        for j in range(len(batchItem[i])):
            item[i,j] = batchItem[i][j]
        #Purchase
        if batchPurchase[i][0] != 0:
            for j in range(len(batchPurchase[i])):
                purchase[i, j] = int(batchPurchase[i][j])
    return item, lengths, torch.LongTensor(np.array(batchTarget)), purchase

def split_index(train_ratio, dev_ratio, total_length):
    index=np.random.permutation(total_length)                  
    #Get split
    trainnum=int(total_length*train_ratio)
    trainindex=index[:trainnum]
    validnum=int(total_length*dev_ratio)
    validindex=index[trainnum:trainnum+validnum]
    testindex=index[trainnum+validnum:total_length]
    return trainindex, validindex, testindex
    
def write_seq(seqlist, filename1, filename2, real=True):
    click=open(os.path.join('',filename1),'a+') 
    target=open(os.path.join('',filename2),'a+') 
    for seq in seqlist:
        for item in seq:
            click.write(str(item) + ' ')
        click.write('\n')
        if real == True:
            target.write('1\n')
        else:
            target.write('0\n')
    click.close()
    target.close()
    
def write_tensor(seq_samples, lengths, filename1, filename2, real=False):
    click=open(os.path.join('',filename1),'a+') 
    target=open(os.path.join('',filename2),'a+') 
    seq_samples = seq_samples.data.cpu().numpy()
    for i in range(len(lengths)):
        for j in range(lengths[i]):
            click.write(str(seq_samples[i,j]) + ' ')
        click.write('\n')
        if real == True:
            target.write('1\n')
        else:
            target.write('0\n')
    click.close()
    target.close()
                    
def gen_fake(generator, trainSample, batch_size, embed_dim, device, max_length=5):
    for stidx in range(0, trainSample.length(), batch_size):
        # prepare batch
        embed_batch, length, _, pur_batch = getBatch(stidx, stidx + batch_size, trainSample, embed_dim) 
        #embed_batch = Variable(embed_batch.to(device))
        embed_batch = embed_batch.to(device)
        pur_batch = pur_batch.to(device)
        with torch.no_grad():
            seq_samples, lengths, _, _ = generator.sample((embed_batch, length), max_length, pur_batch)
        write_tensor(seq_samples, lengths, 'click_gen.txt', 'tar_gen.txt', real=False)
    return seq_samples, lengths
        
def roll_out(generator, discriminator, seq, pre_length, max_length, pur_batch, sample_num, device):
    # seq = (seq, seq_len)
    seq_embed, seq_length = seq
    T = int(np.max(seq_length))
    assert T == seq_embed.size(1)
    init_length = torch.ones(len(seq_length)).type(torch.LongTensor)
    seq_length = torch.from_numpy(seq_length).type(torch.LongTensor)
    reward = torch.zeros(seq_embed.size(0), seq_embed.size(1)).to(device)
    with torch.no_grad():
        for i in range(1, T):
            embed_batch = seq_embed[:, :i]
            lengths = torch.min(init_length*i, seq_length).data.cpu().numpy()
            for j in range(sample_num):
                seq_samples, gen_lengths, _, reward_pur = generator.sample((embed_batch, lengths), max_length, pur_batch)
                seq_samples = seq_samples.to(device)
                output = discriminator((seq_samples, gen_lengths))
                reward[:, i-1] += torch.exp(output[:, 1]) + reward_pur #probability of being real and the reward
            reward[:, i-1] = reward[:, i-1]/sample_num
        reward[:, T-1] = torch.exp(discriminator(seq)[:, 1])
    '''
    #Corresponding places are set to zeros --prob is already set to be 0
    for i in range(len(pre_length)):
        reward[i, :pre_length[i]].zero_()
        reward[i, seq_length[i]:].zero_()
    '''
    return reward          