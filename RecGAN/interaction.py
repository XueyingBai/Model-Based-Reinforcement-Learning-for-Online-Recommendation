import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init    
from generator import Generator
from agent import Agent
    
class Interaction(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, numlabel, feature_vec=None, init_embed=False, model='LSTM'):
        super(Interaction, self).__init__()
        self.batch_size = bsize
        self.n_classes = numlabel
        self.enc_lstm_dim = encod_dim
        self.generator = Generator(self.batch_size, embed_dim, encod_dim, self.n_classes)
        self.agent = Agent(self.batch_size, embed_dim, encod_dim, self.n_classes)
    
    # initialise oracle network with N(0,1)
    # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
    def init_param(self, model, path):
        if model == 'generator':
            self.generator.load_state_dict(torch.load(path))
            self.generator.weight.requires_grad = True
        elif model == 'agent':
            self.agent.load_state_dict(torch.load(path))  
            self.agent.weight.requires_grad = True            
    
    def forward(self, (click_batch, lengths)):
        action, hidden_agent = self.agent.forward((click_batch, lengths))
        _, hidden_gen = self.generator.forward((click_batch, lengths))
        outputk = self.generator.next_click(hidden_gen, action, len(click_batch))
        reward = self.generator.get_reward(hidden_gen)
        #reward = reward.max(1)[1]
        return outputk, hidden_agent, hidden_gen, reward
        
    def step(self, click_batch, hidden_agent, hidden_gen):
        action, hidden_agent = self.agent.step(click_batch, hidden_agent)
        _, hidden_gen = self.generator.forward(click_batch, hidden_gen)
        outputk = self.generator.next_click(hidden_gen, action, len(click_batch))
        reward = self.generator.get_reward(hidden_gen)
        #reward = reward.max(1)[1]
        return outputk, hidden_agent, hidden_gen, reward