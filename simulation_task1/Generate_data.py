import numpy as np
from policy_origin import Policy
from MDP import Environment 
import torch.optim as optim
import torch
from replay_simu import ReplayMemory
import os
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
    outputmodelname="model.pth"
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
    
    #Set environment
    env = Environment(bsize, embed_dim, encod_dim, num_clicks).to(device)
    policy = Policy(bsize, embed_dim_policy, encod_dim_policy, num_clicks-1, recom_number).to(device)
    if load:
        environment = "/home/xb6cf/RL_Recommend/simulation_task1/model_output/environment.model.pth"
        policy_new = "/home/xb6cf/RL_Recommend/seqGan/model_output/seqGan_agent.model.pth"
        env.load_state_dict(torch.load(environment))
        policy.load_state_dict(torch.load(policy_new))
    else:
        env.init_params()
        torch.save(env.state_dict(), os.path.join(outputdir, 'environment.' + outputmodelname))
        #Set initial policy
        policy.init_params()
        torch.save(policy.state_dict(), os.path.join(outputdir, 'orig_policy.' + outputmodelname))
     
    #Generate action and reward sequences
    capacity = 10000
    max_length = 5
    file_action = 'gen_click.txt'
    file_reward = 'gen_reward.txt'
    file_recom = 'gen_action.txt'
    if os.path.isfile(file_action):
        os.remove(file_action)
    if os.path.isfile(file_reward):
        os.remove(file_reward)
    if os.path.isfile(file_recom):
        os.remove(file_recom)
    Replay = ReplayMemory(env, policy, capacity, max_length, num_clicks, recom_number, train = False)
    Replay.gen_sample(bsize)
    Replay.write_sample(file_action, file_reward, file_recom, num_clicks, add_end=False)  
    orig_reward = Replay.rewards.data.cpu().float().sum(1).mean().numpy()
    print('The original reward is: ' + str(orig_reward))
        
    