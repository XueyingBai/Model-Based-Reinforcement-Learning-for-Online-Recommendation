from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import os
#from generator import 
                        
class ReplayMemory(object):
    def __init__(self, env, policy, capacity, max_length, state_num, action_num, end_state, start_states = [None], start_actions = [None]):
        self.env = env
        self.policy = policy
        self.capacity = capacity
        self.max_length = max_length  
        self.action_num = action_num
        self.end_state = end_state 
        self.actions = np.zeros((self.capacity, max_length), dtype=int)
        self.rewards = np.zeros((self.capacity, max_length))
        self.length = np.ones(self.capacity, dtype=int)*max_length
        #States store the state number, not the state feature
        if start_states[0] == None:
            self.states = np.random.randint(state_num, size = self.capacity)  
        else:
            self.states = start_states
            assert len(self.states) == self.capacity  
        #Arrange the actions
        if start_actions[0] == None:
            self.actions[:, 0] = np.random.randint(action_num, size = self.capacity)  
        else:
            self.actions[:, 0] = start_actions
            assert len(self.actions[:, 0]) == self.capacity  
        self.rewards[:, 0] = self.actions[:, 0]
                                       
    def select_action(self, state, cuda=True):        
        state = torch.from_numpy(state).squeeze(1).float()        
        #state = Variable(state)
        if cuda:
            state = state.cuda()           
        action = self.policy(state)
        #x = torch.multinomial(torch.exp(action), 1).squeeze(1)
        #return x.data.cpu().numpy()
        return action.data.max(1)[1].cpu().numpy()
        
    def select_action_testagent(self, action, hidden, gen_type='train', cuda=True):
        action = torch.from_numpy(action).unsqueeze(1)
        if cuda:
            action = action.cuda()
        output, hidden = self.policy.step(action, hidden)
        if gen_type == 'train':
            x = torch.multinomial(torch.exp(output), 1).squeeze(1)
            return x.data.cpu().numpy(), hidden
        else:
            return output.data.max(1)[1].cpu().numpy(), hidden
                
    def next_state(self, state_pre, action):
        state = np.zeros(len(action), dtype=int)
        for i in range(len(action)):
            trans = self.env.SAS_trans(state_pre[i])[action[i]]
            if(np.sum(trans)>0):
                state[i] = np.random.choice(range(self.env.Get_state_num()), 1, replace=False, p=trans)
            else:
                state[i] = self.end_state[0]
        return state
    
    def State_feature(self, state):
        state_fea = []
        for i in range(len(state)):
            state_fea.append(self.env.Get_state(state[i]))
        return np.array(state_fea)
        
    def Get_reward(self, usrfea, state, action):
        reward = []
        for i in range(len(action)):
            #reward.append(self.env.Immediate_reward(usrfea, state[i], action[i]))
            reward.append(self.env.Immediate_reward(action[i]))
        return reward    
    
    #Generate from state                
    def gen_sample(self, usrfea, batch_size):
        for stidx in range(0, self.capacity, batch_size):
            ended = np.array([])
            state_batch = self.states[stidx: stidx + batch_size]
            state_batch_fea = self.State_feature(state_batch)
            for i in range(self.max_length):
                action = self.select_action(state_batch_fea)
                self.actions[stidx: stidx + batch_size, i] = action
                self.rewards[stidx: stidx + batch_size, i] = self.Get_reward(usrfea, state_batch, action)
                state_batch = self.next_state(state_batch, action)
                state_batch_fea = self.State_feature(state_batch)
                #Not adding for the ended states
                index = np.where(state_batch == self.end_state)[0]
                for j in index:
                    if self.length[stidx + j] == self.max_length: # The length hasn't been assigned
                        self.length[stidx + j] = i + 1
                
    #Generate from action
    def gen_sample_testagent(self, usrfea, batch_size, gen_type='train'):
        #During the test, give the first action as the original policy generated
        for stidx in range(0, self.capacity, batch_size):
            ended = np.array([])
            state_batch = self.states[stidx: stidx + batch_size]
            action = self.actions[stidx: stidx + batch_size, 0]   
            #Next state
            state_batch = self.next_state(state_batch, action)
            index = np.where(state_batch == self.end_state)[0]
            self.length[stidx + index] = 1
            #Next action
            action_batch = torch.from_numpy(self.actions[stidx: stidx + batch_size, 0]).unsqueeze(1).cuda()
            output, hidden = self.policy.forward((action_batch, np.ones(len(state_batch), dtype=int)))
            if gen_type == 'train':
                action = torch.multinomial(torch.exp(output), 1).squeeze(1).data.cpu().numpy()
            else:
                action = output.data.max(1)[1].cpu().numpy()
            for i in range(self.max_length-1):
                action = np.where(action == self.action_num, self.actions[stidx: stidx + batch_size, i], action)
                self.actions[stidx: stidx + batch_size, i+1] = action
                self.rewards[stidx: stidx + batch_size, i+1] = self.Get_reward(usrfea, state_batch, action)
                state_batch = self.next_state(state_batch, action)
                #Not adding for the ended states
                index = np.where(state_batch == self.end_state)[0]
                for j in index:
                    if self.length[stidx + j] == self.max_length: # The length hasn't been assigned
                        self.length[stidx + j] = i + 2
                #Generate the next action
                action, hidden = self.select_action_testagent(action, hidden, gen_type)
                                                                
    def write_sample(self, filename_action, filename_reward, num_actions, add_end=True): #write reward and actions
        file_action = open(os.path.join('',filename_action),'a+') 
        file_reward = open(os.path.join('',filename_reward),'a+') 
        for i in range(len(self.actions)):
            for j in range(self.length[i]-1):
                file_action.write(str(self.actions[i,j]) + ' ')
                file_reward.write(str(self.rewards[i,j]) + ' ')
            file_action.write(str(self.actions[i, self.length[i]-1]))
            file_reward.write(str(self.rewards[i, self.length[i]-1]) + '\n')
            if add_end:
                file_action.write(' ' + str(num_actions))
            file_action.write('\n')
        file_action.close()
        file_reward.close()
        
    def __len__(self):
        return len(self.reward)