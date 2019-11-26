import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_embed=0.25):
        super(EmbeddingLayer, self).__init__()
        self.drop = nn.Dropout(drop_embed)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        #self.init_embedding_weights(vocab_size, embed_dim)

    def forward(self, input_variable):
        embedded = self.embedding(input_variable)
        embedded = self.drop(embedded)
        return embedded#, self.embedding.weight.clone()
    '''
    #If there's pretrained feature vectors
    def init_embedding_weights(self, vocab_size, embed_dim):
         
        pretrained_weight = np.zeros([vocab_size, embed_dim], dtype=float)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = True
    '''
    def init_embedding_weights(self, embed):  
        self.embedding.weight = nn.Parameter(embed.weight.data)
        #self.embedding.weight.data.copy_(embed.weight.data)
        #self.embedding.bias.data.copy_(embed.bias.data)
        #self.embedding.weight.requires_grad = True
    
    def print_grad(self, weight = True):
        if weight == True:
            print("Weights:")
            print(self.embedding.weight)
            print((self.embedding.weight!=self.embedding.weight).sum())
        print("Gradients")
        print(self.embedding.weight.grad)
        print((self.embedding.weight.grad!=self.embedding.weight.grad).sum())
        
class Encoder(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, model, n_layers, drop_encoder=0):
        super(Encoder, self).__init__()
        self.bsize = bsize
        self.word_emb_dim =  embed_dim
        self.enc_lstm_dim = encod_dim
        self.dpout_model = drop_encoder
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model=='LSTM':
            self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, n_layers, batch_first = True, bidirectional=False, dropout=self.dpout_model)
        elif self.model=='GRU':
            self.enc_lstm = nn.GRU(self.word_emb_dim, self.enc_lstm_dim, n_layers, batch_first = True, bidirectional=False, dropout=self.dpout_model)
        else:
            self.enc_lstm = nn.RNN(self.word_emb_dim, self.enc_lstm_dim, n_layers, batch_first = True, bidirectional=False, dropout=self.dpout_model, nonlinearity='relu')
                        
    def forward(self, sent_tuple):
        # sent_len [max_len, ..., min_len] (batch) | sent Variable(seqlen x batch x worddim)
        sent, sent_len = sent_tuple      
        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len = sent_len.copy()
        idx_unsort = np.argsort(idx_sort)
        idx_sort = torch.from_numpy(idx_sort).to(self.device)
        sent = sent.index_select(0, Variable(idx_sort))
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len, batch_first=True)
        if self.model=='LSTM':
            sent_output, (h, c) = self.enc_lstm(sent_packed)
        else:
            sent_output, h = self.enc_lstm(sent_packed)
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]
        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
        sent_output = sent_output.index_select(0, Variable(idx_unsort))
        h = h.index_select(1, Variable(idx_unsort)) 
        if self.model=='LSTM':
            c = c.index_select(1, Variable(idx_unsort))
            return sent_output, (h, c)
        else:
            return sent_output, h
            
    def init_hidden(self):  
        h = Variable(torch.zeros(1, self.bsize, self.enc_lstm_dim)).to(self.device)
        if self.model=='LSTM':
            c = Variable(torch.zeros(1, self.bsize, self.enc_lstm_dim)).to(self.device)
            return (h, c)
        return h
        
    def step_cell(self, sent, hidden):
        if self.model=='LSTM':
            sent_output, (h, c) = self.enc_lstm(sent, hidden)
            return sent_output, (h, c)
        else:
            sent_output, h = self.enc_lstm(sent, hidden)
            return sent_output, h
        
    def print_grad(self, weight = True):
        for name, param in self.enc_lstm.named_parameters():
            if param.requires_grad:
                print(name)
                if weight:
                    print("Weights:")
                    print(param)
                    print((param!=param).sum())
                print("Gradients:")
                print(param.grad)
                print((param.grad!=param.grad).sum())