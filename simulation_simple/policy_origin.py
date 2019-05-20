import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init 

class Policy(nn.Module):
    def __init__(self, input_len, num_action):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_len, 128)
        self.action_head = nn.Linear(128, num_action)
        
    def init_params(self):
        for param in self.parameters():
            init.normal(param, 0, 1)

    def forward(self, state):
        #x = F.relu(self.affine1(x))
        x = self.affine1(state)
        #x = self.value_head_1(x)
        action_scores = self.action_head(x)
        action = F.softmax(action_scores, dim=-1)
        return action
        