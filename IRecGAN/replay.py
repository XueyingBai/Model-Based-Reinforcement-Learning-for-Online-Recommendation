from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import os 

#Reward_batch is with the same size as the click_batch
def clear_redundant(reward_batch, lengths, reward = True):
    for i in range(len(lengths)):
        if reward:
            reward_batch[i, lengths[i]:].zero_()
        else:
            reward_batch[i, lengths[i]:]=1
    return reward_batch

#Generate whole click_batch    
def concatenate_tensors(clicks_batch, tgt_rewards_batch, actions_batch, usr_probs_batch, agent_probs_batch, lengths, add_clicks, add_tgt_rewards, add_actions, add_usr_probs, add_agent_probs, add_lengths, recom_length):    
    item_num = len(lengths)
    total_lengths = lengths + add_lengths
    seq_clicks = torch.zeros(item_num, int(np.max(total_lengths))).type(torch.LongTensor).cuda()
    seq_tgt_rewards = torch.zeros(item_num, int(np.max(total_lengths))).type(torch.FloatTensor).cuda()
    seq_actions = torch.zeros(item_num, recom_length, int(np.max(total_lengths))).type(torch.LongTensor).cuda()
    seq_usr_probs = torch.ones(item_num, int(np.max(total_lengths))).cuda()
    seq_agent_probs = torch.ones(item_num, int(np.max(total_lengths))).cuda()
    for i in range(item_num):
        seq_clicks[i, :lengths[i]] = clicks_batch[i, :lengths[i]]
        seq_clicks[i, lengths[i]: total_lengths[i]] = add_clicks[i, :add_lengths[i]]
        seq_tgt_rewards[i, :lengths[i]] = tgt_rewards_batch[i, :lengths[i]]
        seq_tgt_rewards[i, lengths[i]: total_lengths[i]] = add_tgt_rewards[i, :add_lengths[i]]
        seq_usr_probs[i, :lengths[i]] = usr_probs_batch[i, :lengths[i]]
        seq_usr_probs[i, lengths[i]: total_lengths[i]] = add_usr_probs[i, :add_lengths[i]]
        seq_agent_probs[i, :lengths[i]] = agent_probs_batch[i, :lengths[i]]
        seq_agent_probs[i, lengths[i]: total_lengths[i]] = add_agent_probs[i, :add_lengths[i]]
        seq_actions[i, :, :lengths[i]] = actions_batch[i, :, :lengths[i]]
        seq_actions[i, :, lengths[i]: total_lengths[i]] = add_actions[i, :, :add_lengths[i]]
    return seq_clicks, seq_tgt_rewards, seq_actions, seq_usr_probs, seq_agent_probs, total_lengths
                      
class ReplayMemory(object):
    def __init__(self, user, agent, capacity, max_length, action_num, recom_length, evaluate=False):
        self.user = user
        self.agent= agent
        self.capacity = capacity
        self.recom_length = recom_length
        self.sample = 0 #number of the given sequence
        self.max_length = max_length  
        self.action_num = action_num #Including the end symbol
        self.clicks= torch.zeros(capacity, max_length).type(torch.LongTensor).cuda()
        self.tgt_rewards = torch.zeros(capacity, max_length).type(torch.FloatTensor).cuda()
        self.gen_rewards = torch.zeros(capacity, max_length).type(torch.FloatTensor).cuda()
        self.actions = torch.zeros(capacity, recom_length, max_length).type(torch.LongTensor).cuda()
        #self.probs = torch.zeros(capacity, max_length).cuda()
        self.usr_probs = torch.ones(capacity, max_length).cuda()
        self.agent_probs = torch.ones(capacity, max_length).cuda()
        self.lengths = np.ones(self.capacity, dtype=int)
        self.end = action_num - 1 #The end symbol
        self.evaluate = evaluate
            
    def init_click(self, seq, tgt_batch_reward, batch_action):
        # seq : (seq, seq_len)
        # If not given samples, then randomly initialize the first one
        self.clicks[:, 0] = torch.randint(0, self.action_num-1, size = (self.capacity, ), dtype=torch.int).cuda() 
        self.actions[:, :, 0] = torch.ones(self.capacity, self.recom_length).type(torch.LongTensor).cuda() * (self.clicks[:, 0].view(-1,1))
        self.usr_probs[:, 0] = 1
        self.agent_probs[:, 0] = 1
        self.sample = len(seq[1])
        #Put initial click seqnences into the replay batch
        for i in range(self.sample):
            init_len = min(seq[1][i], self.max_length)
            self.clicks[i, :init_len] = seq[0][i][:init_len]
            self.tgt_rewards[i, :init_len] = tgt_batch_reward[i][:init_len]
            self.actions[i, :, :init_len] = batch_action[i][:, :init_len]
            self.lengths[i] = init_len
         
    #Initialized only for the click sampling        
    def init_click_sample(self, seq, batch_reward, batch_action):
        assert self.sample == 0
        self.clicks[:, 0] = seq[0][:, 0]
        self.tgt_rewards[:, 0] = batch_reward[:, 0]
        self.actions[:, :, 0] = batch_action[:, :, 0]
             
    # The agent will give a recommendation list, hidden is the next state               
    def select_action(self, click_batch, lengths, hidden=None, start=False): 
        click_batch = Variable(click_batch)    
        if start:
            outputk, action, hidden = self.agent.forward((click_batch, lengths), self.evaluate)
        else:
            outputk, action, hidden = self.agent.step(click_batch, hidden, self.evaluate)
        #Add EOS
        outputk = torch.cat((outputk, torch.ones(outputk.size(0), 1, requires_grad=True).cuda()), 1)
        return outputk, action, hidden 
    
    # Action is the recommendation list, hidden is from the environment
    def usr_next_pref(self, click_batch, action, lengths, hidden=None, start=False):
        if start:
            enc_out, hidden = self.user.forward((click_batch, lengths))
        else:
            enc_out, hidden = self.user.step(click_batch, hidden)
        #Action add EOS
        outputk = self.user.next_click(enc_out[:, -1, :], torch.cat((action, self.end * torch.ones(action.size(0), 1).type(torch.LongTensor).cuda()), 1), len(click_batch)) #+ 1e-18
        return outputk, enc_out, hidden
    
    def usr_reward(self, enc_out, next_clicks):
        #with torch.no_grad():
        reward, _ = self.user.get_reward(next_clicks, enc_out[:,-1,:].unsqueeze(0))
            #reward = reward.max(1)[1]
            #reward = torch.round(reward)
        return reward
        
    def select_click(self, usr_outputk):
        if self.evaluate:
            x = usr_outputk.max(1)[1].unsqueeze(1)
        else:
            x = torch.multinomial(usr_outputk, 1)
        usr_probs = usr_outputk.gather(1, x.view(-1, 1)).view(-1)
        return x, usr_probs
        
    #Probabilities and rewards for the given sequences
    def pr_given(self, click_batch, action_batch, lengths):
        usr_prob = torch.ones(click_batch.size(0), click_batch.size(1)).cuda()
        agent_prob = torch.ones(click_batch.size(0), click_batch.size(1)).cuda()
        usr_prob[:, 0] = 1 #for the first given click, torch.ones(click_batch.size(0))
        agent_prob[:, 0] = 1
        #tgt_reward = torch.zeros(click_batch.size(0), click_batch.size(1)).type(torch.LongTensor).cuda()
        #gen_reward = torch.ones(click_batch.size(0), click_batch.size(1)).type(torch.LongTensor).cuda()
        sample_max_length = int(np.max(lengths))
        for i in range(sample_max_length-1): 
            click_current = click_batch[:, i].view(-1, 1)
            action_next = action_batch[:, :, i+1]
            if i == 0:
                outputk, action, hidden_agent = self.select_action(click_current, np.ones(len(click_current), dtype=int), None, True) 
                usr_outputk, enc_out, hidden_usr = self.usr_next_pref(click_current, action_next, np.ones(len(click_current), dtype=int), None, True)     
            else:
                #If click_current == eos, replace it with 0
                click_current_agent = click_current.clone()
                click_current_agent[click_current_agent == self.end] = 0
                outputk, action, hidden_agent = self.select_action(click_current_agent, None, hidden_agent) 
                usr_outputk, enc_out, hidden_usr = self.usr_next_pref(click_current, action_next, None, hidden_usr)     
            #(a_t, c_t, r_t)
            agent_prob[:, i+1] = outputk.gather(1, click_batch[:, i+1].view(-1, 1)).view(-1) #+ 1e-10 
            usr_prob[:, i+1] = usr_outputk.gather(1, click_batch[:, i+1].view(-1, 1)).view(-1) #+ 1e-18 
        return usr_prob, agent_prob 
    
    #Simulate samples for a batch, click_batch is the given part of clicks
    def cpr_add(self, click_batch, lengths):
        item_num = len(lengths) #actual batch size
        local_max_length = int(np.max(self.max_length - lengths))
        add_lengths = np.ones(item_num, dtype=int)*local_max_length
        add_usr_probs = torch.ones(item_num, local_max_length).cuda()
        add_agent_probs = torch.ones(item_num, local_max_length).cuda()
        add_clicks = torch.zeros(item_num, local_max_length).type(torch.LongTensor).cuda()  
        add_actions = torch.zeros(item_num, self.recom_length, local_max_length).type(torch.LongTensor).cuda()
        add_tgt_rewards = torch.zeros(item_num, local_max_length).type(torch.FloatTensor).cuda()       
        for i in range(local_max_length):
            if i == 0:
                outputk, action, hidden_agent = self.select_action(click_batch, lengths, None, True)
                usr_outputk, enc_out, hidden_usr = self.usr_next_pref(click_batch, action, lengths, None, True)
            else:
            #outputk, action, hidden_agent = self.select_action(click_current, None, hidden_agent)
                #no EOS
                click_current_agent = click_current.clone()
                click_current_agent[click_current_agent == self.end] = 0
                outputk, action, hidden_agent = self.select_action(click_current_agent, None, hidden_agent)
                usr_outputk, enc_out, hidden_usr = self.usr_next_pref(click_current, action, None, hidden_usr)
            click_current, usr_probs = self.select_click(usr_outputk)
            #Calculate tgt rewards
            add_tgt_rewards[:, i] = self.usr_reward(enc_out, click_current)
            add_usr_probs[:, i] = usr_probs
            add_agent_probs[:, i] = outputk.gather(1, click_current.view(-1, 1)).view(-1)
            add_clicks[:, i] = click_current.view(-1)
            add_actions[:, :, i] = action
            index = np.where(click_current.data.cpu().numpy() == self.end)[0]
            for j in index:
                if add_lengths[j] == local_max_length: # The length hasn't been assigned
                    add_lengths[j] = i 
                    #print("Meet the end at " + str(i))
                    #print(add_lengths[j])    
        return add_clicks, add_tgt_rewards, add_actions, add_usr_probs, add_agent_probs, add_lengths
        
    def cpr_sample(self, click_batch, tgt_reward_batch, action_batch, lengths):
        usr_probs_batch, agent_probs_batch = self.pr_given(click_batch, action_batch, lengths)
        add_clicks, add_tgt_rewards, add_actions, add_usr_probs, add_agent_probs, add_lengths = self.cpr_add(click_batch, lengths)
        clicks, tgt_rewards, actions, usr_probs, agent_probs, lengths = concatenate_tensors(click_batch, tgt_reward_batch, action_batch, usr_probs_batch, agent_probs_batch, lengths, add_clicks, add_tgt_rewards, add_actions, add_usr_probs, add_agent_probs, add_lengths, self.recom_length)
        #Cut redundant lengths
        lengths = np.minimum(lengths, self.max_length)
        return clicks[:, :self.max_length], tgt_rewards[:, :self.max_length], actions[:, :, :self.max_length], usr_probs[:, :self.max_length], agent_probs[:, :self.max_length], lengths
                
    #Generate next clicks            
    def gen_sample(self, batch_size, rollout = True, discriminator = None):
        #Put given samples into the replay buffer
        for stidx in range(0, self.sample, batch_size):
            click_batch = self.clicks[stidx: stidx + batch_size].clone()
            action_batch = self.actions[stidx: stidx + batch_size].clone().cuda()
            length_batch = self.lengths[stidx: stidx + batch_size]
            usr_probs_batch, agent_probs_batch = self.pr_given(click_batch, action_batch, length_batch)
            self.usr_probs[stidx: stidx + batch_size, :usr_probs_batch.size(1)] = usr_probs_batch  
            self.agent_probs[stidx: stidx + batch_size, :agent_probs_batch.size(1)] = agent_probs_batch 
            self.gen_rewards[stidx: stidx + batch_size, 1:] = 1.0 #tgt rewards are given in the initialization
        #Simulate other sequences 
        for stidx in range(self.sample, self.capacity, batch_size):
            click_batch = self.clicks[stidx: stidx + batch_size].clone().cuda()
            length_batch = self.lengths[stidx: stidx + batch_size]
            tgt_reward_batch = self.tgt_rewards[stidx: stidx + batch_size].clone().cuda()
            action_batch = self.actions[stidx: stidx + batch_size].clone().cuda()
            click_temp, tgt_rewards_temp, actions_temp, usr_probs_temp, agent_probs_temp, self.lengths[stidx: stidx + batch_size] = self.cpr_sample(click_batch, tgt_reward_batch, action_batch, length_batch)
            real_length = int(np.max(self.lengths[stidx: stidx + batch_size]))
            self.clicks[stidx: stidx + batch_size][:, :real_length], self.tgt_rewards[stidx: stidx + batch_size][:, :real_length], self.actions[stidx: stidx + batch_size][:, :, :real_length], self.usr_probs[stidx: stidx + batch_size][:, :real_length], self.agent_probs[stidx: stidx + batch_size][:, :real_length] = click_temp, tgt_rewards_temp, actions_temp, usr_probs_temp, agent_probs_temp
            #Roll-out operation            
            if rollout == True:
                dis_reward = self.roll_out(self.clicks[stidx: stidx + batch_size], self.tgt_rewards[stidx: stidx + batch_size], self.actions[stidx: stidx + batch_size], self.lengths[stidx: stidx + batch_size], discriminator) 
                self.gen_rewards[stidx: stidx + batch_size] = dis_reward.type(torch.FloatTensor).cuda()
                self.tgt_rewards[stidx: stidx + batch_size] = self.tgt_rewards[stidx: stidx + batch_size]
        #Clear redundant rewards and probs
        self.gen_rewards = clear_redundant(self.gen_rewards, self.lengths)
        self.tgt_rewards = clear_redundant(self.tgt_rewards, self.lengths)
        self.usr_probs = clear_redundant(self.usr_probs, self.lengths, False)
        self.agent_probs = clear_redundant(self.agent_probs, self.lengths, False)
    
    #Only for simulated sequences        
    def roll_out(self, click_batch, tgt_reward_batch, action_batch, lengths, discriminator, sample_num = 10):
        T = int(np.max(lengths))
        dis_reward = torch.zeros(click_batch.size(0), click_batch.size(1)).cuda()
        init_length = np.ones(len(lengths), dtype=int)
        with torch.no_grad():
            for i in range(1, T):   
                lengths_batch = np.minimum(init_length*i, lengths)
                for j in range(sample_num):
                    gen_clicks, gen_tgt_rewards, gen_actions, gen_usr_probs, gen_agent_probs, gen_lengths = self.cpr_sample(click_batch[:, :i], tgt_reward_batch[:, :i], action_batch[:, :, :i], lengths_batch)
                    gen_tgt_rewards = gen_tgt_rewards.type(torch.FloatTensor).cuda()
                    output = discriminator((gen_clicks, gen_lengths), gen_tgt_rewards, gen_actions)
                    dis_reward[:, i-1] += torch.exp(output[:, 1]) #probability of being real and the reward
                    #tgt_reward[:, i] += gen_tgt_rewards
                dis_reward[:, i-1] = dis_reward[:, i-1]/sample_num
            dis_reward[:, T-1] = torch.exp(discriminator((click_batch, lengths), tgt_reward_batch, action_batch))[:, 1] #,reward_batch.type(torch.FloatTensor).cuda())[:, 1]
        return dis_reward
        
    def __len__(self):
        return self.capacity
