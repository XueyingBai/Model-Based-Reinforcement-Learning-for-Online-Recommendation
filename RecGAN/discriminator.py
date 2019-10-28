import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init    
from nn_layer import EmbeddingLayer, Encoder

class Discriminator(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, embed_dim_policy, encod_dim_policy, numlabel, rec_num, numclass=2, feature_vec=None, init_embed=False, model='LSTM'):
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
        #self.encoder = eval(self.encoder_type)(self.batch_size, embed_dim + 1, self.enc_lstm_dim, self.model, 1)
        self.encoder = eval(self.encoder_type)(self.batch_size, embed_dim + rec_num + 1, self.enc_lstm_dim, self.model, 1)
        #self.encoder = eval(self.encoder_type)(self.batch_size, 3, self.enc_lstm_dim, self.model, 1)
        self.enc2out = nn.Linear(self.enc_lstm_dim, self.n_classes)
        self.rec2enc = nn.Linear(embed_dim * rec_num, rec_num)
        #self.rec2enc = nn.Linear(embed_dim, 1)
        #self.emb2enc = nn.Linear(embed_dim, 1)
         
    def forward(self, seq, reward, rec):
        # seq : (seq, seq_len)
        seq_em, seq_len= seq
        seq_em = self.embedding(seq_em)
        # rescale the recommendation list
        rec = rec.permute(0,2,1)
        
        #Calculate the real rec length
        #rec_num = (rec != 0).sum(dim=2).type(torch.FloatTensor).cuda()
         
        #rec_num = rec.sum(dim=2).type(torch.FloatTensor).cuda()
         
        rec_em = rec.contiguous().view(-1, rec.size(2))
        rec_em = self.embedding(rec_em) 
        #Take the max element at each embedding
        #rec_em = torch.max(rec_em, 1, keepdim=True)[0].view(rec.size(0), rec.size(1), -1)
        rec_em = rec_em.view(rec.size(0), rec.size(1), -1)
        #print(rec_em.size())
        
        #rec_em = rec_em/rec_num.unsqueeze(2).expand_as(rec_em)
        
        rec_em = self.rec2enc(rec_em)
        #seq_em = self.emb2enc(seq_em)
        #Concatenate with the reward
        seq_em = torch.cat((seq_em, rec_em, reward.unsqueeze(2)), 2)
        #seq_em = torch.cat((seq_em, reward.unsqueeze(2)), 2)
        #seq_em = torch.cat((seq_em, reward.unsqueeze(2)), 2)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder((seq_em, seq_len))
        else:
            enc_out, h = self.encoder((seq_em, seq_len))
         
        # Mean pooling
        seq_len = torch.FloatTensor(seq_len.copy()).unsqueeze(1).cuda()
        #print(enc_out.size())
        enc_out = torch.sum(enc_out, 1).squeeze(1)
        enc_out = enc_out / seq_len.expand_as(enc_out)
         
        # Extract the last hidden layer
        #output = self.enc2out(enc_out)#batch*hidden
        output = self.enc2out(h.squeeze(0))#batch*hidden
        output = F.log_softmax(output, dim=1) #batch*n_classes
        #print(output)
        return output
     
    '''
    def forward(self, seq, reward, rec):
        # seq : (seq, seq_len)
        seq_em, seq_len= seq
        seq_em = self.embedding(seq_em)
        #Concatenate with the reward
        seq_em = torch.cat((seq_em, reward.unsqueeze(2)), 2)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder((seq_em, seq_len))
        else:
            enc_out, h = self.encoder((seq_em, seq_len))
        # Extract the last hidden layer
        output = self.enc2out(h.squeeze(0)) #batch*hidden
        output = F.log_softmax(output, dim=1) #batch*n_classes
        return output
    ''' 