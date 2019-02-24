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
    
class ReplayMemory(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.value = []
        self.reward = []
        self.next_value = []
        self.action = []
        self.monte_value = []
        self.log_prob_value = []
        #self.st_ac_num = np.zeros(len(env.S), len(env.A))
        self.b_size = batch_size
        
    def select_action_DQN(self, st, model, target=False, cuda=True):
        state = torch.from_numpy(st).float()        
        state = Variable(state)
        if cuda:
            state = state.cuda()   
        state_value = model(state)
        #greedy
        if target:
            saved_value = state_value.max(1)[0].detach()
            if cuda:
                action = state_value.max(1)[1].data.cpu().numpy()[0]
            else:
                action = state_value.max(1)[1].data.numpy()[0]
        else:
            #eps-greedy
            global steps_done
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if sample > eps_threshold:
                #m = Categorical(probs)
                #action = m.sample()
                if cuda:
                    action = state_value.max(1)[1].data.cpu().numpy()[0]
                else:
                    action = state_value.max(1)[1].data.numpy()[0]
                saved_value = state_value.max(1)[0]
            else:
                action = random.randrange(state_value.size(1))
                saved_value = state_value[:, action]
        log_prob = torch.log(saved_value)
        return action, saved_value, log_prob
    
    def select_action_AC(self, st, model, target=False, cuda=True):
        state = torch.from_numpy(st).float()        
        state = Variable(state)
        if cuda:
            state = state.cuda()   
        probs, state_value = model(state)
        #m = Categorical(probs)
        if target:
            saved_value = state_value.detach()
        else:
            saved_value = state_value
        if cuda:
            action = probs.max(1)[1].data.cpu().numpy()[0]
        else:
            action = probs.max(1)[1].data.numpy()[0]
        log_prob = torch.log(probs.max(1)[0])
        return action, saved_value, log_prob

    def push_train(self, env, model, target_model, usr_fea, select_action):
        """Saves a transition."""
        st = random.sample(range(env.Get_state_num()),1)[0]
        each_value=[]
        each_reward=[]
        each_next_value=[]
        each_monte_value=[]
        each_log_value=[]
        '''
        each_monte_num=[]
        each_monte_prev=[]
        '''
        action, state_value, log_prob=select_action(env.Get_state(st), model)
        reward=env.Immediate_reward(usr_fea, st, action)
        trans=env.SAS_trans(st)[action]
        while len(each_value) < self.capacity and trans.sum() > 0:
            each_value.append(state_value)
            each_reward.append(reward)
            each_log_value.append(log_prob)
            '''
            #Monte-carlo G value
            _, state_value_cur=select_action(env.Get_state(st), target_model, True)
            self.st_ac_num[st, action] += 1
            each_monte_prev.append(state_value_cur)
            each_monte_num.append(self.st_ac_num)
            '''
            #TD(0) value
            st=np.random.choice(range(env.Get_state_num()), 1, replace=False, p=trans)
            _, state_value_target, _ = select_action(env.Get_state(st), target_model, True)                
            each_next_value.append(state_value_target)
            
            action, state_value, log_prob=select_action(env.Get_state(st), model)
            reward=env.Immediate_reward(usr_fea, st, action)
            trans=env.SAS_trans(st)[action]
        #For monte carlo method (TD lambda)
        '''
        R = 0
        for i in range(len(each_reward)-1, -1, -1):
            R = R + env.gamma * each_reward[i]
            #num = each_monte_num[i]
            #prev = each_monte_prev[i]
            #each_monte_value.insert(0, prev + 1/num*(R-prev))
            each_monte_value.insert(0, R)
        '''
        each_monte_value = preturn_lambda(each_reward, each_next_value, env.gamma, lamb=0.5, k=2)
        #return each_action, each_reward, each_value
        return each_value, each_reward, each_next_value, each_monte_value, each_log_value
    
    def push_eval(self, env, model, baseline, usr_fea, select_action):
        """Saves a transition."""
        st = random.sample(range(env.Get_state_num()),1)[0]
        each_reward=[]
        each_base_reward=[]
        #each_action=[]
        action, state_value, log_prob =select_action(env.Get_state(st), model, True)
        reward=env.Immediate_reward(usr_fea, st, action)
        trans=env.SAS_trans(st)[action]
        while len(each_reward) < self.capacity and trans.sum()>0:
            #transitions of the baseline
            base_action = baseline[st]
            base_reward = env.Immediate_reward(usr_fea, st, base_action)
            base_trans = env.SAS_trans(st)[base_action]
            #each_action.append(action)
            each_reward.append(reward)
            each_base_reward.append(base_reward)
            
            action, state_value, log_prob =select_action(env.Get_state(st), model, True)
            reward=env.Immediate_reward(usr_fea, st, action)
            trans=env.SAS_trans(st)[action]
            
        return np.sum(each_reward)-np.sum(each_base_reward)
    
    def sample_train(self, env, model, target_model, usr_fea_vec):
        usr_id=0
        for i in range(self.b_size):
            #usr_id=random.sample(range(len(usr_fea_vec)),1)
            each_value, each_reward, each_next_value, each_monte_value, each_log_value=self.push_train(env, model, target_model, usr_fea_vec[usr_id], self.select_action_DQN)
            while len(each_value) == 0:
                each_value, each_reward, each_next_value, each_monte_value, each_log_value=self.push_train(env, model, target_model, usr_fea_vec[usr_id], self.select_action_DQN)
            self.value.append(each_value)
            self.reward.append(each_reward)
            self.next_value.append(each_next_value)
            self.monte_value.append(each_monte_value)
            self.log_prob_value.append(each_log_value)
                
    def sample_eval(self, env, model, baseline, usr_fea_vec):
        usr_id=0
        loss=0
        for i in range(self.b_size):
            #usr_id=random.sample(range(len(usr_fea_vec)),1)
            value_loss=self.push_eval(env, model, baseline, usr_fea_vec[usr_id], self.select_action_DQN)
            #self.action.append(each_action)
            loss += value_loss
        return loss

    def __len__(self):
        return len(self.reward)