import numpy as np
import os
from replay_simu import ReplayMemory 
import torch
import pickle
from policy_origin import Policy
from MDP import Environment 
from util import get_args, get_optimizer

def GetFileReward(filename, discount=1):
    total_reward = 0
    file_action = open(os.path.join('',filename), 'r')
    rewardline = file_action.readlines()
    reward = np.zeros(len(rewardline))
    for i in range(len(rewardline)):
        words=rewardline[i].rstrip().split(" ")
        for j in range(len(words)):
            reward[i] += float(words[j])
            total_reward += float(words[j])
    mean_reward = total_reward/len(rewardline)
    return reward, mean_reward
         
def Eval(policy_new="/home/IRecGAN/model_output/agent.pickle"):
    #Absolute route
    filename1='/home/simulation_task1/gen_reward.txt'
    _, mean_orig = GetFileReward(filename1)
     
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #The environment
    env_path = "/home/simulation_task1/model_output/environment.pickle"
    num_clicks=100
    recom_length=20
    args = get_args()
    bsize=args.batch_size
    #Load environment
    env = torch.load(env_path)
    #Set policy
    policy = torch.load(policy_new)
    #Generate the start states
    capacity = 10000
    max_length = 5
    filename1='/home/simulation_task1/gen_reward.txt'
    Replay = ReplayMemory(env, policy, capacity, max_length, num_clicks, recom_length, evaluate=True)
    Replay.gen_sample(bsize)
    #Replay.write_sample('click.txt', 'reward.txt', 'action.txt', num_clicks, add_end=False) 
    mean_new = Replay.rewards.data.cpu().float().sum(1).mean().numpy()
    #print(GetFileReward('reward.txt')[1])
    print("The original reward is:" + str(mean_orig))
    print("The optimal reward is:" + str(mean_new))
    return mean_orig, mean_new
    
if __name__ == '__main__':
    mean_orig, mean_optim = Eval(policy_new)
    
