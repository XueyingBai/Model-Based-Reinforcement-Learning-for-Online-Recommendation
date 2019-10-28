import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init    
from nn_layer import EmbeddingLayer, Encoder

class Generator(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, numlabel, n_layers, feature_vec=None, init=False, model='LSTM'):
        super(Generator, self).__init__()
        # classifier
        self.batch_size = bsize
        self.nonlinear_fc = False
        self.n_classes = numlabel
        self.enc_lstm_dim = encod_dim
        self.encoder_type = 'Encoder'
        self.n_layers = n_layers
        #self.end = self.n_classes-1
        self.model = model
        self.gamma = 0.9
        self.embedding=EmbeddingLayer(numlabel, embed_dim, 0)
        '''
        if init:
            self.embedding.init_embedding_weights(feature_vec, embed_dim)
        '''
        self.encoder = eval(self.encoder_type)(self.batch_size, embed_dim, self.enc_lstm_dim, self.model, self.n_layers, 0)
        #self.enc2out = nn.Linear(self.enc_lstm_dim, self.n_classes)
        self.enc2out = nn.Linear(self.enc_lstm_dim, embed_dim)
        #self.enc2pur = nn.Linear(self.enc_lstm_dim, 2)
        self.enc2rewd = nn.Linear(self.enc_lstm_dim, embed_dim)
        #self.enc2out = torch.randn(embed_dim, self.enc_lstm_dim, requires_grad = True).cuda()
        #self.outbias = torch.randn(embed_dim, 1, requires_grad = True).cuda()
        #self.outbias2 = torch.randn(self.n_classes, requires_grad = True).cuda()
        #self.enc2rewd = torch.randn(embed_dim, self.enc_lstm_dim, requires_grad = True).cuda()
        #self.rewdbias = torch.randn(embed_dim, 1, requires_grad = True).cuda()
        #self.rewdbias2 = torch.randn(1, requires_grad = True).cuda()
        if init:
            self.init_params()
    
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
            
    def step(self, seq, hidden):
        seq_em = self.embedding(seq)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder.step_cell(seq_em, hidden)
            return enc_out, (h, c) 
        else:
            enc_out, h = self.encoder.step_cell(seq_em, hidden)
            return enc_out, h
            
    #Generate the current reward        
    def get_reward(self, seq, enc_out):
        #_, h = self.step(seq, hidden)
        #with torch.no_grad():
        seq_em = self.embedding(seq)
        #vec = torch.matmul(seq_em, self.enc2rewd)
        #vec_bias = torch.matmul(seq_em, self.rewdbias)
        #reward = torch.matmul(vec, enc_out.permute(1, 2, 0)) + vec_bias
        #reward = reward + self.rewdbias2.expand(reward.size())
        #print((enc_out!=enc_out).sum())
        vec = self.enc2rewd(enc_out)
        reward_logit = torch.sum(seq_em.permute(1,0,2) * vec, dim = 2).squeeze()
        #reward = F.log_softmax(reward, dim=1)
        reward = torch.sigmoid(reward_logit) 
        #print(reward)
        return reward, reward_logit
    
    #Next item without recommendation
    def next_simple(self, enc_out):
        embed_weight = self.embedding.embedding.weight.clone().permute(1,0)
        output = torch.matmul(self.enc2out(enc_out), embed_weight)
        output = F.softmax(output, dim=1)
        return output 
        
    #Next click with recommendation    
    def next_click(self, enc_out, rec_list, real_batch):
        embed_weight = self.embedding.embedding.weight.clone().permute(1,0)
        vec = self.enc2out(enc_out)
        #vec = torch.matmul(embed_weight, self.enc2out)
        #vec_bias = torch.matmul(embed_weight, self.outbias)
        vec = torch.matmul(vec, embed_weight)
        #vec = (torch.matmul(vec, enc_out.permute(1,0)) + vec_bias).permute(1,0) 
        #vec = vec + self.outbias2.expand(vec.size())
        mask = torch.zeros(real_batch, self.n_classes).cuda()
        mask.scatter_(1, rec_list, 1.)
        
        #output = torch.exp(vec) #+ 1e-12
        #print(output.size())
        output = F.softmax(vec * mask, dim=1)
        #output = F.softmax(vec, dim=1)
        outputk = output * mask
        #Normalization
        #outputk = outputk/(torch.sum(outputk, dim=1, keepdim=True) + 1e-12) + 1e-12
        outputk = F.normalize(outputk, p=1)
        #output = torch.min(output, torch.ones(output.size()).cuda())
        #print(outputk)
        #print((outputk!=outputk).sum())
        return outputk
        
    def value(self, reward_batch):
        with torch.no_grad():
            value_batch = reward_batch.clone()
            for i in range(1, reward_batch.size(1)):
                value_batch[:, -(i+1)] = self.gamma * value_batch[:, -i] + value_batch[:, -(i+1)]
        return value_batch