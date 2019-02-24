import torch.nn as nn
import torch
import numpy as np

########################Network structure for the user behavior model#############################
class EmbeddingLayer(nn.Module):
    """Embedding class which includes only an embedding layer."""

    def __init__(self, input_size, embed_dim):
        """"Constructor of the class"""
        super(EmbeddingLayer, self).__init__()
        #self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embed_dim)

    def forward(self, input_variable):
        """"Defines the forward computation of the embedding layer."""
        embedded = self.embedding(input_variable)
        #embedded = self.drop(embedded)
        return embedded
    
    def init_embedding_weights(self, feature_vec, embed_dim):
        """Initialize weight parameters for the embedding layer."""
        pretrained_weight = np.zeros([len(feature_vec), embed_dim], dtype=float)
        for i in feature_vec:
            pretrained_weight[i] = feature_vec[i]
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = True
        
class Encoder(nn.Module):    
    def __init__(self, bsize, embed_dim, encod_dim, model):
        super(Encoder, self).__init__()
        self.bsize = bsize
        self.word_emb_dim =  embed_dim
        self.enc_lstm_dim = encod_dim
        self.pool_type = 'mean'
        self.dpout_model = 0
        self.use_cuda = True
        
        if model=='LSTM':
            #self.enc_lstm = nn.LSTM(self.word_emb_dim+1, self.enc_lstm_dim, 1, bidirectional=False, dropout=self.dpout_model)
            self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1, bidirectional=False, dropout=self.dpout_model)
        else:
            self.enc_lstm = nn.RNN(self.word_emb_dim+1, self.enc_lstm_dim, 1, bidirectional=False, dropout=self.dpout_model, nonlinearity='relu')
            #self.enc_lstm = nn.GRU(self.word_emb_dim+1, self.enc_lstm_dim, 1, bidirectional=False, dropout=self.dpout_model)
            
    def forward(self, sent_tuple, time):
        # sent_len [max_len, ..., min_len] (batch) | sent Variable(seqlen x batch x worddim)
        sent, sent_len = sent_tuple        
        
        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.use_cuda else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, Variable(idx_sort))
        #time = time.index_select(1, Variable(idx_sort)).mul(10e5).float()
        #time_sent=torch.cat([sent,time.unsqueeze(2).expand(*time.size(),1)],2)
        time_sent = sent
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(time_sent, sent_len)
        
        sent_output = self.enc_lstm(sent_packed)[0] #seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]
        
        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.use_cuda else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))
        return sent_output
        
class PPNet(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, numlabel, feature_vec, init_embed=False, model='RNN_relu'):
        super(PPNet, self).__init__()
        # classifier
        self.nonlinear_fc = False
        self.fc_dim = 512
        self.n_classes = numlabel
        self.enc_lstm_dim = encod_dim
        self.encoder_type = 'Encoder'
        self.dpout_fc = 0
        self.embedding=EmbeddingLayer(numlabel, embed_dim)
        if init_embed:
            self.embedding.init_embedding_weights(feature_vec, embed_dim)
        self.encoder = eval(self.encoder_type)(bsize, embed_dim, encod_dim,model)
        
        self.inputdim = self.enc_lstm_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.inputdim, self.n_classes),
            nn.LogSoftmax(2)
            )            
        self.intensity = nn.Sequential(
            nn.Linear(self.inputdim, 1)
            )
         
        #self.w = nn.Parameter(torch.Tensor([10e-6]))
        #self.w = 1e-6
        self.w = nn.Parameter(torch.rand(1))
         
    def forward(self, seq, time):#, tgt_time):
        # seq : (seq, seq_len)
        seq_em, seq_len= seq
        seq_em2 = self.embedding(seq_em)
        h = self.encoder((seq_em2, seq_len),time)[-1,:,:].unsqueeze(0)
        #n = self.noise(h)
        outmark = self.classifier(h).squeeze()
        #outmark = torch.mean(n+outmark, 1, True)
        '''
        outinten_1 = self.intensity(h).squeeze(2)
        #outinten_2 = self.w.expand_as(tgt_time)*tgt_time
        outinten_2 = self.w*tgt_time
        term1 = torch.add(outinten_1, outinten_2)
        term2 = torch.div(torch.exp(outinten_1), self.w)
        term3 = torch.div(torch.exp(term1), self.w)
        f= term1 + term2 - term3      
        '''            
        f =0
        return outmark, f
    
    def encode(self, seq, time):
        seq_em, seq_len= seq
        seq_em2 = self.embedding(seq_em)
        h = self.encoder((seq_em2, seq_len),time)[-1,:,:].squeeze()
        return h