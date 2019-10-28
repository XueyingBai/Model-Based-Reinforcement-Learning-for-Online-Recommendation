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
         
def Eval():
    #filename1='/home/RecGAN/gen_reward2.txt'
    filename2='/home/RecGAN/reward_gen.txt' 
    filename1='/home/simulation_task1/gen_reward.txt'
    _, mean_orig = GetFileReward(filename1)
    _, mean_pred = GetFileReward(filename2)
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #The environment
    environment = "/home/simulation_task1/model_output/environment.model.pth"
    #policy_new = "/home/RecGAN/model_output/seqGan_agent.model.pth"
    policy_new = "/home/simulation_task1/model_output/model_free_simple.policy.pth"
    #policy_new = "/home/simulation_task1/model_output/orig_policy.model.pth"
    num_clicks=100
    recom_length=20
    args = get_args()
    bsize=args.batch_size
    embed_dim=args.embed_dim
    encod_dim=args.nhid
    embed_dim_policy = args.embed_dim_policy
    encod_dim_policy=args.nhid_policy
    init_embed=args.init_embed
    model_type=args.model
    seedkey=args.seed
    #Load environment
    env = Environment(bsize, embed_dim, encod_dim, num_clicks).to(device)
    env.load_state_dict(torch.load(environment))
    #env = torch.load(environment)
    #Set policy
     
    policy = Policy(bsize, embed_dim_policy, encod_dim_policy, num_clicks-1, recom_length).to(device)
    #print(policy)
    policy.load_state_dict(torch.load(policy_new))
     
    #policy = torch.load("/home/simulation_task1/model_output/orig_policy.model.pickle")
    #Generate the start states
    capacity = 10000
    max_length = 5
    start_action = GetStartAction(filename1)
    capacity = len(start_action)
    '''
    file_click = 'gen_click1.txt'
    file_reward = 'gen_reward1.txt'
    file_action = 'gen_action1.txt'
    fileclick = 'click_gen_real1.txt'
    if os.path.isfile(file_action):
        os.remove(file_action)
    if os.path.isfile(file_reward):
        os.remove(file_reward)
    if os.path.isfile(file_click):
        os.remove(file_click)
    '''
    Replay = ReplayMemory(env, policy, capacity, max_length, num_clicks, recom_length, start_action, train=False)
    Replay.gen_sample(bsize)
    #Replay.write_sample(file_click, file_reward, file_action, num_clicks, add_end=False) 
    mean_new = Replay.rewards.data.cpu().float().sum(1).mean().numpy()
     
    #filename2='gen_reward.txt'
    #EvalFile(filename1, file_reward)
    #mean_orig, mean_optim = EvalFile(filename1, file_reward)
    print("The original reward is:" + str(mean_orig))
    print("The optimal reward is:" + str(mean_new))
    return mean_orig, mean_new#, mean_optim_pre
    
if __name__ == '__main__':
    #mean_orig, mean_optim, mean_optim_pre = Eval()
    mean_orig, mean_optim = Eval()
    
