import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init    
from nn_layer_simu import EmbeddingLayer, Encoder
    
class Environment(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, numlabel, feature_vec=None, init=False, model='LSTM'):
        super(Environment, self).__init__()
        '''
        # Use drop out to get a simpler model
        self.dropout = 0.5
        self.drop = nn.Dropout(self.dropout)
        '''
        # classifier
        self.batch_size = bsize
        self.nonlinear_fc = False
        self.n_classes = numlabel
        self.enc_lstm_dim = encod_dim
        self.encoder_type = 'Encoder'
        self.model = model
        self.embedding = EmbeddingLayer(numlabel, embed_dim)
        self.gamma = 0.9
        '''
        if init:
            self.embedding.init_embedding_weights(feature_vec, embed_dim)
        '''
        self.encoder = eval(self.encoder_type)(self.batch_size, embed_dim, self.enc_lstm_dim, self.model)
        self.enc2out = nn.Linear(self.enc_lstm_dim, embed_dim)
        self.enc2rewd = nn.Linear(self.enc_lstm_dim, embed_dim)
        self.end = numlabel-1
    
    # initialise oracle network with N(0,1)
    # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
    def init_params(self):
        for param in self.parameters():
            init.normal_(param, 0, 1)
             
    def forward(self, seq):
        # seq : (seq, seq_len)
        seq_em, seq_len= seq
        seq_em = self.embedding(seq_em)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder((seq_em, seq_len))
            return enc_out, (h, c)
        else:
            enc_out, h = self.encoder((seq_em, seq_len))
            return enc_out, h
                    
    def step(self, click, hidden):
        seq_em = self.embedding(click)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder.step_cell(seq_em, hidden)
            return enc_out, (h, c) 
        else:
            enc_out, h = self.encoder.step_cell(seq_em, hidden)
            return enc_out, h
            
    def next_click(self, enc_out, rec_list, real_batch): 
        embed_weight = self.embedding.embedding.weight.clone().permute(1,0)
        vec = self.enc2out(enc_out)
        vec = torch.matmul(vec, embed_weight)
        mask = torch.zeros(real_batch, self.n_classes).cuda()
        mask.scatter_(1, rec_list, 1.) 
        output = F.softmax(vec * mask, dim=1)
        outputk = output * mask
        #Normalization
        outputk = F.normalize(outputk, p=1)
        return outputk 
    
    def reward(self, seq, enc_out):
        seq_em = self.embedding(seq)
        vec = self.enc2rewd(enc_out)
        reward_logit = torch.sum(seq_em.permute(1,0,2) * vec, dim = 2).squeeze()
        reward = torch.sigmoid(reward_logit) 
        return reward, reward_logit
    
    def value(self, reward_batch):
        with torch.no_grad():
            value_batch = reward_batch.clone()
            for i in range(1, reward_batch.size(1)):
                value_batch[:, -(i+1)] = self.gamma * value_batch[:, -i] + value_batch[:, -(i+1)]
        return value_batch
