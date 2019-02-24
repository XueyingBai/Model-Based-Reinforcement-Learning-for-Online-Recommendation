from torch.distributions import Categorical
from collections import namedtuple
from torch.autograd import Variable
import torch.nn as nn
import torch
import math
import numpy as np
import random

steps_done = 0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
###################################################
#k-step td-lambda 
###################################################
def get_w(lamb, k, K):
    if k<K:
        return (1-lamb**k)*(lamb**(k*(k-1)/2))
    else:
        return (lamb**(K*(K-1)/2))
    
def preturn_lambda(each_reward, each_next_value, gamma, lamb=0.5, k=3):
    K = len(each_reward)
        #get g_k
        #g_k = np.zeros(k+1)
    R = np.zeros(len(each_reward))
    if len(each_reward)>0:
        R[-1] = each_reward[-1]
        for i in range(k, 0, -1):
            for j in range(len(each_reward)-i):
                g_k = each_reward[j+i] + each_next_value[j+i].data.cpu().numpy()
                for n in range(j+i-1, j-1, -1):
                    g_k = each_reward[n] + gamma*g_k
                R[j] += get_w(lamb, i, K)*g_k
    return R.tolist()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'preference', 'reward'))
                            
class ReplayMemory(object):
    def __init__(self, capacity=200):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        #Forget the previous ones

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        
def getState_preference(model, length, input_item, input_time):
    with torch.no_grad():
        state = model.encode((input_item, length), input_time)
        preference, _ = model((input_item, length), input_time)
        #next_click = output.data.max(2)[1].cpu().numpy()
    return state, preference
    
def getClick(state_value, preference, k=20):
    state_value = torch.exp(state_value * preference)
    sorted, indices = torch.sort(state_value)
    state_value[indices[k:]] = 0
    final_score = state_value/torch.sum(state_value)
    action =  final_score.data.max(0)[1].cpu().numpy()
    '''
    m = Categorical(final_score)
    action = m.sample().data.to("cpu").numpy()
    '''
    return action

def getReward(action, purchase):
    if action == purchase:
        reward = [1]
    else:
        reward = [-1]       
    return reward 
     
def select_click(state, preference, evaluate = False, k=20):
    global steps_done
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) *         math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #The model is the training model
    #state = torch.from_numpy(state).float().to(device)       
    with torch.no_grad():
        if sample > eps_threshold or evaluate:
            state_value = policy_net(state)    
        else:    
            state_value = torch.ones(numlabel).to(device)
    #return state_value
    return getClick(state_value, preference, k)