import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init    
from nn_layer import EmbeddingLayer, Encoder
    
class Agent(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, numlabel, n_layers, recom = 100, feature_vec=None, init=False, model='LSTM'):
        super(Agent, self).__init__()
        # classifier
        self.batch_size = bsize
        self.nonlinear_fc = False
        self.n_classes = numlabel
        self.enc_lstm_dim = encod_dim
        self.encoder_type = 'Encoder'
        self.model = model
        self.gamma = 0.9
        self.n_layers = n_layers
        self.recom = recom #Only top 10 items are selected
        self.embedding=EmbeddingLayer(numlabel, embed_dim)
        '''
        if init:
            self.embedding.init_embedding_weights(feature_vec)
        '''
        self.encoder = eval(self.encoder_type)(self.batch_size, embed_dim, self.enc_lstm_dim, self.model, self.n_layers)
        self.enc2out = nn.Linear(self.enc_lstm_dim, self.n_classes)
        if init:
            self.init_params()
    
    # initialise oracle network with N(0,1)
    # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
    def copy_weight(self, feature_vec):
            self.embedding.init_embedding_weights(feature_vec)
            
    def init_params(self):
        for param in self.parameters():
            init.normal_(param, 0, 1)
             
    def forward(self, seq, evaluate=False):
        # seq : (seq, seq_len)
        seq_em, seq_len= seq
        seq_em = self.embedding(seq_em)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder((seq_em, seq_len))
        else:
            enc_out, h = self.encoder((seq_em, seq_len))
        output = self.enc2out(enc_out[:, -1, :])#batch*hidden
        output = F.softmax(output, dim=1)
        #indices is with size of batch_size*self.recom
        if evaluate:
            _, indices = torch.topk(output, self.recom, dim = 1, sorted = True)
        else:
            indices = torch.multinomial(output, self.recom)
        if self.model == 'LSTM':
            return output, indices, (h, c)
        else:
            return output, indices, h
        
    def step(self, click, hidden, evaluate=False):
        seq_em = self.embedding(click)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder.step_cell(seq_em, hidden)
        else:
            enc_out, h = self.encoder.step_cell(seq_em, hidden)
        output = self.enc2out(enc_out[:, -1, :]) #batch*hidden
        output = F.softmax(output, dim=1)
        if not evaluate:
            indices = torch.multinomial(output, self.recom)
        else:
            _, indices = torch.topk(output, self.recom, dim = 1, sorted = True)
        # Only select from the top k
        if self.model == 'LSTM':
            return output, indices, (h, c)
        else:
            return output, indices, h
        
