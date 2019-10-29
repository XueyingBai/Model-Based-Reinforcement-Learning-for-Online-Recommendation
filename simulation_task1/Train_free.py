import numpy as np
from policy_origin import Policy
from MDP import Environment 
import torch.optim as optim
import torch
from replay_simu import ReplayMemory
import os
from util import get_args, get_optimizer
from train import train_gen_pg_each
import matplotlib
from evaluate import Eval
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def save_plot(epoch_num, step, rewards, filepath):
    """Generate and save the plot"""
    fig, ax = plt.subplots()    
    ax.plot(range(0, epoch_num + 1, step), rewards)#,'.')
    ax.plot(range(0, epoch_num + 1, step), np.ones(len(range(0, epoch_num + 1, step)))*rewards[0], 'r')
    fig.savefig(filepath)
    plt.close(fig)  # close the figure     
    
def train(optims, max_epoch, policy, bsize, env, num_clicks, recom_number, max_length, origin_reward, capacity):
    outputdir="model_output"
    policy_new=os.path.join(outputdir, 'model_free_simple.pickle') 
    
    #weight = torch.FloatTensor(numlabel).fill_(1)
    optim_fn, optim_params=get_optimizer(optims)
    optimizer = optim_fn(filter(lambda p: p.requires_grad, policy.parameters()), **optim_params)
    
    n_epochs=max_epoch
    max_reward = 0
    epoch = 1
    best_model = None
    rewards = [origin_reward]
    while epoch <= n_epochs:
        _ = train_gen_pg_each(policy, env, epoch, optimizer, num_clicks, recom_number, max_length, bsize, total_size = capacity)
        print('saving policy at epoch {0}'.format(epoch))
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        torch.save(policy, policy_new)
        #Eval the new policy
        _, mean_reward = Eval(policy_new)
        rewards.append(mean_reward)
        # save model        
        if mean_reward >= max_reward:
            best_model = policy
            max_reward = mean_reward
        epoch += 1
    return best_model, rewards, max_reward

if __name__ == '__main__':
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputdir="model_output"
    policy_new=os.path.join(outputdir, 'model_free_simple.pickle') 
    #Define the environment
    num_clicks=100
    recom_number = 20
    args = get_args()
    optim = args.optim
    bsize = args.batch_size
    #Load environment
    env_path = "./model_output/environment.pickle"
    env = torch.load(env_path)
    #Load initial policy
    policy_path = "./model_output/orig_policy.pickle"
    policy = torch.load(policy_path)
    torch.save(policy, policy_new)
    
    #Training for model-free reinforcement learning
    max_epoch = 200
    max_length = 5
    capacity = 10000
    origin_reward, _ = Eval(policy_new)
    _, rewards, max_reward = train(optim, max_epoch, policy, bsize, env, num_clicks, recom_number, max_length, origin_reward, capacity)        
    #Plot rewards
    save_plot(max_epoch, 1, rewards, 'rewards_test.png')
    #Write rewards
    f=open('rewards_model_free.txt','ab+')
    np.savetxt(f,rewards)
    f.close()
