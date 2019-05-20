import matplotlib.pyplot as plt
import numpy as np
import os
from replay import ReplayMemory
import torch
import pickle

def GetFileReward(filename, discount=0.99):
    total_reward = 0
    file_action = open(os.path.join('',filename), 'r')
    rewardline = file_action.readlines()
    reward = np.zeros(len(rewardline))
    for i in range(len(rewardline)):
        words=rewardline[i].rstrip().split(" ")
        for j in range(len(words)):
            reward[i] += float(words[j])*(discount**j)
            total_reward += float(words[j])
    mean_reward = total_reward/len(rewardline)
    return reward, mean_reward
    
def GetStartAction(filename):
    file_action = open(os.path.join('',filename), 'r')
    actionline = file_action.readlines()
    action = np.zeros(len(actionline))
    for i in range(len(actionline)):
        words=actionline[i].rstrip().split(" ")
        action[i] = int(words[0])
    return action

def EvalFile(filename1, filename2):
    reward0, mean0 = GetFileReward(filename1)
    reward1, mean1 = GetFileReward(filename2)
    print("The mean reward for original policy is:")
    print(mean0)
    print("The mean reward for optimized policy is:")
    print(mean1)
         
if __name__ == '__main__':      
    filename1='gen_reward.txt'
    filename2='/home/xb6cf/RL_Recommend/seqGan/click_gen_2.txt'
    fileaction='gen_action.txt'
    EvalFile(filename1, filename2)
     
    num_states=40
    fea_states=11
    num_actions=20
    fea_actions=6
    usr_num=20
    usr_fea=5
    start_action = GetStartAction(fileaction)
    #Load the action model
    agent_model = torch.load("/home/xb6cf/RL_Recommend/seqGan/model_output/seqGan_gen1.model.pickle")
    #Load the environment
    with open('environment.pickle', 'rb') as f:
        env = pickle.load(f)  
    with open('usr_fea_vec.pickle', 'rb') as f:
        usr_fea_vec = pickle.load(f) 
    with open('end_state.pickle', 'rb') as f:
        end_state = pickle.load(f) 
    with open('start_states.pickle', 'rb') as f:
        start_states = pickle.load(f) 
           
    capacity = 10000
    max_length = 5
    batch_size = 128 
    file_action = 'agent_action.txt'
    file_reward = 'agent_reward.txt'
    if os.path.isfile(file_action):
        os.remove(file_action)
    if os.path.isfile(file_reward):
        os.remove(file_reward)
    Replay = ReplayMemory(env, agent_model, capacity, max_length, num_states, num_actions, end_state, start_states, start_action)
    usr_fea_vec = np.zeros(usr_fea)
    Replay.gen_sample_testagent(usr_fea_vec[0], batch_size, 'test')
    Replay.write_sample(file_action, file_reward, num_actions, add_end=False)     
     
    filename1='gen_reward.txt'
    filename2='agent_reward.txt'
    EvalFile(filename1, filename2)