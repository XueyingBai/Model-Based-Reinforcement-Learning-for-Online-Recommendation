import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#Actor Critic NN
class ActCritic(nn.Module):
    def __init__(self, input_len, action_len):
        super(ActCritic, self).__init__()
        self.affine1 = nn.Linear(input_len, 128)
        self.action_head = nn.Linear(128, action_len)
        self.fca1 = nn.Linear(action_len,128)
        self.value_head = nn.Linear(128+128, 1)
        #self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        #x = F.relu(self.affine1(x))
        x = self.affine1(x)
        action_scores = self.action_head(x)
        action_expand = self.fca1(action_scores)
        x_concat = torch.cat((x,action_expand),dim=1)
        state_values = self.value_head(x_concat)
        return F.softmax(action_scores, dim=-1), state_values
        
class DQN(nn.Module):
    def __init__(self, input_len, action_len):
        super(DQN, self).__init__()
        self.affine1 = nn.Linear(input_len, 128)
        self.value_head = nn.Linear(128, action_len)

    def forward(self, x):
        #x = F.relu(self.affine1(x))
        x = self.affine1(x)
        #x = self.value_head_1(x)
        state_values = self.value_head(x)
        return state_values