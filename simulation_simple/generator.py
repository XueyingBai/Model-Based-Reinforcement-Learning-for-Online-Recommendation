import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init    
from nn_layer import EmbeddingLayer, Encoder

'''
def pur_reward(output, pur_batch):
    #print(output[:, 0].size())
    #print(pur_batch.size())
    reward = torch.zeros(pur_batch.size(0)).cuda()
    for i in range(pur_batch.size(1)):
        reward += output[:, 0].long().eq(pur_batch[:, i].data.long()).float()  
    return reward 
'''

def pur_reward(output, pur_batch):
    reward = output[:, 0].float() 
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