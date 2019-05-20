import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init    
from nn_layer import EmbeddingLayer, Encoder
    
class Environment(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, numlabel, feature_vec=None, init_embed=False, model='LSTM'):
        super(Environment, self).__init__()
        # classifier
        self.batch_size = bsize
        self.nonlinear_fc = False
        self.n_classes = numlabel
        self.enc_lstm_dim = encod_dim
        self.encoder_type = 'Encoder'
        self.model = model
        self.recom = 10 #Only top 20 items are selected
        self.embedding=EmbeddingLayer(numlabel, embed_dim)
        if init_embed:
            self.embedding.init_embedding_weights(feature_vec, embed_dim)
        self.encoder = eval(self.encoder_type)(self.batch_size, embed_dim, self.enc_lstm_dim, self.model)
        self.enc2out = nn.Linear(self.enc_lstm_dim, self.n_classes)
        self.enc2pur = nn.Linear(self.enc_lstm_dim, self.n_classes)
        self.end = numlabel-1
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
            
    def next_click(self, h, rec_list, real_batch):
        if self.model == 'LSTM':
            output = torch.exp(self.enc2out(h[0].squeeze(0))) #batch*hidden -> batch*n_classes
        else:
            output = torch.exp(self.enc2out(h.squeeze(0)))
        outputk = torch.zeros(real_batch, self.n_classes).cuda().scatter_(1, rec_list, output)
        outputk = outputk/torch.sum(outputk, dim=1, keepdim=True)
        return outputk
            
    def reward(self, h, outputk):
        if self.model == 'LSTM':
            purchase = self.enc2pur(h[0].squeeze(0)) #batch*hidden -> batch*n_classes
        else:
            purchase = self.enc2pur(h.squeeze(0)) 
        reward = torch.sum(purchase*outputk, 1) 
        return reward