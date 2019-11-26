import numpy as np
from policy_origin import Policy
from MDP import Environment 
import torch.optim as optim
import torch
from replay_simu import ReplayMemory
import os
import sys
sys.path.append('../IRecGAN')
from util import get_args, get_optimizer
'''
import pickle
def save(parameter, filename):
    with open(filename, 'wb') as f:
        pickle.dump(parameter, f)
'''
if __name__ == '__main__':
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputdir="model_output"
    #Define the environment
    num_clicks=100
    recom_number = 20
    args = get_args()
    bsize=args.batch_size
    embed_dim=args.embed_dim
    encod_dim=args.nhid
    embed_dim_policy = args.embed_dim_policy
    encod_dim_policy=args.nhid_policy
    init_embed=args.init_embed
    model_type=args.model
    seedkey=args.seed
    optim = args.optim
    load = args.load 
    
    #print(torch.load("/home/xb6cf/RL_Recommend/seqGan/model_output/agent.pickle"))
    #Set environment
    if load:
        #Absolute route
        environment = "/home/simulation_task1/model_output/environment.pickle"
        policy_new = "/home/IRecGAN/model_output/agent.pickle"
        env = torch.load(environment)
        policy = torch.load(policy_new)
    else:
        env = Environment(bsize, embed_dim, encod_dim, num_clicks).to(device)
        policy = Policy(bsize, embed_dim_policy, encod_dim_policy, num_clicks-1, recom_number).to(device)
        env.init_params()
        torch.save(env, os.path.join(outputdir, 'environment.pickle'))
        #Set initial policy
        policy.init_params()
        torch.save(policy, os.path.join(outputdir, 'orig_policy.pickle'))
     
    #Generate action and reward sequences
    capacity = 10000
    max_length = 5
    #Absolute route
    file_action = '/home/simulation_task1/gen_click.txt'
    file_reward = '/home/simulation_task1/gen_reward.txt'
    file_recom = '/home/simulation_task1/gen_action.txt'
    #for i in range(5):
    if os.path.isfile(file_action):
        os.remove(file_action)
    if os.path.isfile(file_reward):
        os.remove(file_reward)
    if os.path.isfile(file_recom):
        os.remove(file_recom)
    Replay = ReplayMemory(env, policy, capacity, max_length, num_clicks, recom_number, evaluate=True)
    Replay.gen_sample(bsize)
    Replay.write_sample(file_action, file_reward, file_recom, num_clicks, add_end=False)  
    orig_reward = Replay.rewards.data.cpu().float().sum(1).mean().numpy()
    print('\nThe original reward is: ' + str(orig_reward))

    
