import numpy as np
from policy_origin import Policy
from MDP import Environment 
import torch.optim as optim
import torch
from replay import ReplayMemory
import os
from util import get_args, get_optimizer
import pickle
def save(parameter, filename):
    with open(filename, 'wb') as f:
        pickle.dump(parameter, f)
        
if __name__ == '__main__':
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputdir="model_output"
    outputmodelname="model.pickle"
    #Define the environment
    num_clicks=20
    args = get_args()
    bsize=args.batch_size
    embed_dim=args.embed_dim
    encod_dim=args.nhid
    init_embed=args.init_embed
    model_type=args.model
    seedkey=args.seed
    #Set environment
    env = Environment(bsize, embed_dim, encod_dim, num_clicks).to(device)
    env.init_params()
    torch.save(env, os.path.join(outputdir, 'environment.' + outputmodelname))
    #Set policy
    policy = Policy(bsize, embed_dim, encod_dim, num_clicks).to(device)
    policy.init_params()
    torch.save(policy, os.path.join(outputdir, 'orig_policy.' + outputmodelname))
      
    #Generate the start states
    capacity = 10000
    max_length = 5
    file_action = 'gen_click.txt'
    file_reward = 'gen_reward.txt'
    if os.path.isfile(file_action):
        os.remove(file_action)
    if os.path.isfile(file_reward):
        os.remove(file_reward)
    Replay = ReplayMemory(env, policy, capacity, max_length, num_clicks)
    Replay.gen_sample(bsize)
    Replay.write_sample(file_action, file_reward, num_clicks, add_end=False)  
      
        
    