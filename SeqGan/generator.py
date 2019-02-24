import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init    
from nn_layer import EmbeddingLayer, Encoder

def pur_reward(output, pur_batch):
    reward = torch.zeros(pur_batch.size(0)).cuda()
    for i in range(pur_batch.size(1)):
        reward += output[:, 0].long().eq(pur_batch[:, i].data.long()).float()     
    return reward 
    
class Generator(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, numlabel, feature_vec=None, init_embed=False, model='LSTM'):
        super(Generator, self).__init__()
        # classifier
        self.batch_size = bsize
        self.nonlinear_fc = False
        self.n_classes = numlabel
        self.enc_lstm_dim = encod_dim
        self.encoder_type = 'Encoder'
        self.model = model
        self.embedding=EmbeddingLayer(numlabel, embed_dim)
        if init_embed:
            self.embedding.init_embedding_weights(feature_vec, embed_dim)
        self.encoder = eval(self.encoder_type)(self.batch_size, embed_dim, self.enc_lstm_dim, self.model)
        self.enc2out = nn.Linear(self.enc_lstm_dim, self.n_classes)
        #self.init_params()
    
    # initialise oracle network with N(0,1)
    # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
    def init_params(self):
        for param in self.parameters():
            init.normal(param, 0, 1)
            
    def forward(self, seq):
        # seq : (seq, seq_len)
        seq_em, seq_len= seq
        seq_em = self.embedding(seq_em)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder((seq_em, seq_len))
        else:
            enc_out, h = self.encoder((seq_em, seq_len))
        # Extract the last hidden layer
        output = self.enc2out(h.squeeze(0)) #batch*hidden
        output = F.log_softmax(output, dim=1) #batch*n_classes
        if self.model == 'LSTM':
            return output, (h, c)
        else:
            return output, h
        
    def step(self, seq, hidden):
        seq_em = self.embedding(seq)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder.step_cell(seq_em, hidden)
        else:
            enc_out, h = self.encoder.step_cell(seq_em, hidden)
        output = self.enc2out(h.squeeze(0)) #batch*hidden
        output = F.log_softmax(output, dim=1)
        if self.model == 'LSTM':
            return output, (h, c)
        else:
            return output, h
    
    def sample(self, seq, max_length, pur_batch):
        # seq : (seq, seq_len)
        item_num = len(seq[1]) #actual batch size
        add_length = np.ones(item_num).astype(int)*max_length
        pre_length = seq[1]
        max_pre = int(np.max(seq[1]))
        assert max_pre == seq[0].size(1)
        add_probs = torch.zeros(item_num, max_length).cuda()
        add_samples = torch.zeros(item_num, max_length).type(torch.LongTensor).cuda()  
        reward = torch.zeros(item_num).cuda() #The reward for the whole sentence
        discount = 1  #If not 1, can be problematic. Remain to solve         
        #First step
        output, hidden = self.forward(seq)
        x = torch.multinomial(torch.exp(output), 1)
        reward += pur_reward(x, pur_batch)
        for i in range(max_length):
            #Add new clicks, record the probability
            add_probs[:, i] = output.gather(1, x.view(-1, 1)).view(-1)
            add_samples[:, i] = x.view(-1).data
            #find if there's x==0, then record the sentence length
            index = (x == 0).nonzero().view(-1)
            for j in index:
                if add_length[j] == max_length: # The length hasn't been assigned
                    add_length[j] = i+1
            #Generate next clicks
            output, hidden = self.step(x, hidden)
            x = torch.multinomial(torch.exp(output), 1) #May be problemetic 
            reward += (discount ** i)*pur_reward(x, pur_batch)                
        #Generate new seqbatch
        lengths = pre_length+add_length
        seq_samples = torch.zeros(item_num, int(np.max(lengths))).type(torch.LongTensor)
        prob = torch.zeros(item_num, int(np.max(lengths)))
        for i in range(len(pre_length)):
            seq_samples[i, :pre_length[i]] = seq[0][i, :pre_length[i]]
            seq_samples[i, pre_length[i]: lengths[i]] = add_samples[i, :add_length[i]]
            prob[i, pre_length[i]: lengths[i]] = add_probs[i, :add_length[i]]
        return seq_samples, lengths, prob, reward
        
        