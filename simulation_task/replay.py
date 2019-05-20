from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import os
#from generator import 
                        
class ReplayMemory(object):
    def __init__(self, env, policy, capacity, max_length, action_num, start_clicks = [None]):
        self.env = env
        self.policy = policy
        self.capacity = capacity
        self.max_length = max_length  
        self.action_num = action_num
        self.clicks= np.zeros((self.capacity, max_length), dtype=int)
        self.rewards = np.zeros((self.capacity, max_length))
        self.length = np.ones(self.capacity, dtype=int)*max_length
        #Arrange the actions
        if start_clicks[0] == None:
            self.clicks[:, 0] = np.random.randint(action_num-1, size = self.capacity)  
        else:
            self.clicks[:, 0] = start_actions
            assert len(self.actions[:, 0]) == self.capacity
    
    # The agent will give a recommendation list, hidden is the next state                                   
    def select_action(self, click_batch, hidden=None, start=False):     
        if start:
            action, hidden = self.policy.forward((click_batch, np.ones(len(click_batch), dtype=int)))
        else:
            action, hidden = self.policy.step(click_batch, hidden)
        return action, hidden
    
    # The input should only be the click list    
    def select_action_testagent(self, click_batch, lengths):
        action = self.policy.forward((click_batch, lengths))
        return action
        
    # Action is the recommendation list, hidden is from the environment
    def next_click(self, click_batch, action, hidden=None, start=False):
        if start:
            _, hidden = self.env.forward((click_batch, np.ones(len(click_batch), dtype=int)))
        else:
            _, hidden = self.env.step(click_batch, hidden)
        outputk = self.env.next_click(hidden, action, len(click_batch))
        reward = self.env.reward(hidden, outputk)
        x = torch.multinomial(outputk, 1)
        return x, hidden, reward
             
    #Generate from state                
    def gen_sample(self, batch_size):
        for stidx in range(0, self.capacity, batch_size):
            #Start clicks
            click_batch = self.clicks[stidx: stidx + batch_size]
            click_batch = torch.from_numpy(click_batch).cuda()
            action, hidden_agent = self.select_action(click_batch, None, True)
            click_batch, hidden_env, reward = self.next_click(click_batch, action, None, True)
            index = np.where(click_batch.data.cpu().numpy() == self.env.end)[0]
            self.length[stidx + index] = 1
            for i in range(self.max_length-1):
                self.clicks[stidx: stidx + batch_size, i+1] = click_batch.squeeze(1).data.cpu().numpy()
                self.rewards[stidx: stidx + batch_size, i] = reward.data.cpu().numpy()
                # Agent
                action, hidden_agent = self.select_action(click_batch, hidden_agent)
                # Environment
                click_batch, hidden_env, reward = self.next_click(click_batch, action, hidden_env)
                #Not adding for the ended states
                index = np.where(click_batch.data.cpu().numpy() == self.env.end)[0]
                for j in index:
                    if self.length[stidx + j] == self.max_length: # The length hasn't been assigned
                        self.length[stidx + j] = i + 2
            self.rewards[stidx: stidx + batch_size, self.max_length-1] = reward.data.cpu().numpy() 
           
    def gen_sample_test(self, batch_size):
        for stidx in range(0, self.capacity, batch_size):
            #Start clicks
            click_batch = self.clicks[stidx: stidx + batch_size]
            click_batch = torch.from_numpy(click_batch).cuda()
            action = self.select_action_testagent((click_batch, np.ones(len(click_batch), dtype=int)))
            click_batch, hidden_env, reward = self.next_click(click_batch, action, None, True)
            index = np.where(nextclick == self.env.end)[0]
            self.length[stidx + index] = 1
            for i in range(self.max_length-1):
                self.clicks[stidx: stidx + batch_size, i+1] = click_batch.squeeze(1).data.cpu().numpy()
                self.rewards[stidx: stidx + batch_size, i] = reward.data.cpu().numpy()
                # Agent
                clicklist_batch = torch.from_numpy(self.clicks).cuda()
                action = self.select_action_testagent((clicklist_batch, (i+2)*np.ones(len(clicklist_batch), dtype=int)))
                # Environment
                click_batch, hidden_env, reward = self.next_click(click_batch, action, hidden_env)
                #Not adding for the ended states
                index = np.where(click_batch.data.cpu().numpy() == self.env.end)[0]
                for j in index:
                    if self.length[stidx + j] == self.max_length: # The length hasn't been assigned
                        self.length[stidx + j] = i + 2
            self.rewards[stidx: stidx + batch_size, self.max_length-1] = reward.data.cpu().numpy() 
                                              
    def write_sample(self, filename_click, filename_reward, num_items, add_end=True): #write reward and actions
        file_click = open(os.path.join('',filename_click),'a+') 
        file_reward = open(os.path.join('',filename_reward),'a+') 
        for i in range(len(self.clicks)):
            for j in range(self.length[i]-1):
                file_click.write(str(self.clicks[i,j]) + ' ')
                file_reward.write(format(self.rewards[i,j], '.4f') + ' ')
            file_click.write(str(self.clicks[i, self.length[i]-1]))
            file_reward.write(format(self.rewards[i, self.length[i]-1], '.4f') + '\n')
            if add_end:
                file_click.write(' ' + str(num_items))
            file_click.write('\n')
        file_click.close()
        file_reward.close()
        
    def __len__(self):
        return len(self.reward)