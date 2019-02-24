import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init    
from nn_layer import EmbeddingLayer, Encoder

class Discriminator(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, numlabel, numclass=2, feature_vec=None, init_embed=False, model='LSTM'):
        super(Discriminator, self).__init__()
        # classifier
        self.batch_size = bsize
        self.nonlinear_fc = False
        self.n_classes = numclass
        self.enc_lstm_dim = encod_dim
        self.encoder_type = 'Encoder'
        self.model = model
        self.embedding=EmbeddingLayer(numlabel, embed_dim)
        if init_embed:
            self.embedding.init_embedding_weights(feature_vec, embed_dim)
        self.encoder = eval(self.encoder_type)(self.batch_size, embed_dim, self.enc_lstm_dim, self.model)
        self.enc2out = nn.Linear(self.enc_lstm_dim, self.n_classes)
            
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
        return output