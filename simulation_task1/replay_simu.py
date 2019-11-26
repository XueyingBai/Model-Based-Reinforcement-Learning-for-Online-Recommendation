from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import os
#from generator import 
                        
class ReplayMemory(object):
    def __init__(self, env, policy, capacity, max_length, action_num, recom_length = 10, start_clicks = [None], evaluate=False):
        self.evaluate = evaluate
        self.env = env
        self.policy = policy
        self.capacity = capacity
        self.max_length = max_length  
        self.action_num = action_num
        self.recom_num = recom_length
        self.clicks = torch.zeros(self.capacity, max_length).type(torch.LongTensor).cuda()
        self.probs = torch.ones(self.capacity, max_length).type(torch.FloatTensor).cuda()
        self.rewards = torch.zeros(self.capacity, max_length).type(torch.FloatTensor).cuda()
        self.actions = torch.zeros(self.capacity, self.recom_num, max_length).type(torch.LongTensor).cuda()
        self.length = np.ones(self.capacity, dtype=int)*max_length
        #Arrange the actions
        if start_clicks[0] == None:
            self.clicks[:, 0] = torch.from_numpy(np.random.randint(action_num-1, size = self.capacity))  
        else:
            self.clicks[:, 0] = torch.from_numpy(start_clicks)
            assert len(self.clicks[:, 0]) == self.capacity
        self.actions[:, :, 0] = torch.ones(self.capacity, self.recom_num).type(torch.LongTensor).cuda() * self.clicks[:, 0].view(-1,1)#(torch.from_numpy(self.clicks[:, 0]).view(-1,1))
    
    # The agent will give a recommendation list, hidden is the next state                                   
    def select_action(self, click_batch, hidden=None, start=False):     
        if start:
            outputk, action, hidden = self.policy.forward((click_batch, np.ones(len(click_batch), dtype=int)), self.evaluate)
        else:
            outputk, action, hidden = self.policy.step(click_batch, hidden, self.evaluate)
        #Add EOS
        outputk = torch.cat((outputk, torch.ones(outputk.size(0), 1, requires_grad=True).cuda()), 1)
        return outputk, action, hidden
    
    # The input should only be the click list    
    def select_action_testagent(self, click_batch, lengths):
        _, action = self.policy.forward((click_batch, lengths), self.evaluate)
        return action
        
    # Action is the recommendation list, hidden is from the environment
    def next_click(self, click_batch, action, hidden=None, start=False):
        with torch.no_grad():
            if start:
                enc_out, hidden = self.env.forward((click_batch, np.ones(len(click_batch), dtype=int)))
            else:
                enc_out, hidden = self.env.step(click_batch, hidden)
            #outputk = self.env.next_click(hidden, action, len(click_batch))
            outputk = self.env.next_click(enc_out[:,-1,:], torch.cat((action, self.env.end * torch.ones(action.size(0), 1).type(torch.LongTensor).cuda()), 1), len(click_batch))
            #x = torch.multinomial(outputk, 1)
            x = outputk.max(1)[1].unsqueeze(1)
            reward, _ = self.env.reward(x, enc_out[:,-1,:].unsqueeze(0))
            reward = torch.round(reward)
            #print(x)
        return x, hidden, reward
             
    #Generate from state                
    def gen_sample(self, batch_size):
        for stidx in range(0, self.capacity, batch_size):
            #Start clicks
            action_tmp = []
            click_batch = self.clicks[stidx: stidx + batch_size].clone()
            #click_batch = torch.from_numpy(click_batch).cuda()
            outputk, action, hidden_agent = self.select_action(click_batch, None, True)
            click_batch, hidden_env, reward = self.next_click(click_batch, action, None, True)
            index = np.where(click_batch.data.cpu().numpy() == self.env.end)[0]
            self.length[stidx + index] = 1
            for i in range(self.max_length-1):
                self.clicks[stidx: stidx + batch_size, i+1] = click_batch.squeeze(1)#.data.cpu().numpy()
                self.probs[stidx: stidx + batch_size, i+1] = outputk.gather(1, click_batch.view(-1, 1)).view(-1)
                #print(self.probs.requires_grad)
                self.rewards[stidx: stidx + batch_size, i+1] = reward.data#.cpu().numpy()
                self.actions[stidx: stidx + batch_size, :, i+1] = action
                # Agent
                #If click_current == eos, replace it with 0
                click_current_agent = click_batch.clone()
                click_current_agent[click_current_agent == self.env.end] = 0
                outputk, action, hidden_agent = self.select_action(click_current_agent, hidden_agent)
                # Environment
                click_batch, hidden_env, reward = self.next_click(click_batch, action, hidden_env)
                #Not adding for the ended states
                index = np.where(click_batch.data.cpu().numpy() == self.env.end)[0]
                for j in index:
                    if self.length[stidx + j] == self.max_length: # The length hasn't been assigned
                        self.length[stidx + j] = i + 2
            #self.rewards[stidx: stidx + batch_size, self.max_length-1] = reward#.data.cpu().numpy() 
            #self.actions[stidx: stidx + batch_size, :, self.max_length] = action
            #Clear redundant inputs
            for k in range(len(self.length[stidx: stidx + batch_size])):
                l = self.length[stidx + k]
                self.rewards[stidx + k, l:] = 0
                self.probs[stidx+k, l:] = 1
                
    def write_action_line(self, file_action, action_tensor, sep = ' '):
        action = action_tensor.data.cpu().numpy()
        for i in range(len(action)-1):
            file_action.write(str(action[i])+',')
        file_action.write(str(action[-1])+sep)
                                      
    def write_sample(self, filename_click, filename_reward, filename_action, num_items, add_end=True): #write reward and actions
        clicks = self.clicks.data.cpu().numpy()
        rewards = self.rewards.data.cpu().numpy()
        file_click = open(os.path.join('',filename_click),'a+') 
        file_reward = open(os.path.join('',filename_reward),'a+') 
        file_action = open(os.path.join('',filename_action),'a+')
        for i in range(len(self.clicks)):
            for j in range(self.length[i]-1):
                file_click.write(str(clicks[i,j]) + ' ')
                file_reward.write(format(rewards[i,j], '.0f') + ' ')
                self.write_action_line(file_action, self.actions[i,:,j])
            file_click.write(str(clicks[i, self.length[i]-1]))
            file_reward.write(format(rewards[i, self.length[i]-1], '.0f') + '\n')
            self.write_action_line(file_action, self.actions[i,:,self.length[i]-1], '')
            if add_end:
                file_click.write(' ' + str(num_items))
                file_action.write(' ')
                self.write_action_line(file_action, (torch.ones(self.recom_num).type(torch.LongTensor) * num_items).cuda(), '')
            file_action.write('\n')
            file_click.write('\n')
        file_click.close()
        file_reward.close()
        file_action.close()
        
    def __len__(self):
        return len(self.reward)