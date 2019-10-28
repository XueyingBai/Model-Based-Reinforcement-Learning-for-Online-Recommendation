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
    
def train(optims, max_epoch, policy, env, num_clicks, recom_number, max_length, origin_reward, capacity):
    outputdir="model_output"
    outputmodelname="policy.pth" 
    
    #weight = torch.FloatTensor(numlabel).fill_(1)
    optim_fn, optim_params=get_optimizer(optims)
    optimizer = optim_fn(filter(lambda p: p.requires_grad, policy.parameters()), **optim_params)
    
    n_epochs=max_epoch
    max_reward = 0
    epoch = 1
    best_model = None
    rewards = [origin_reward]
    while epoch <= n_epochs:
        _ = train_gen_pg_each(policy, env, epoch, optimizer, num_clicks, recom_number, max_length, total_size = capacity)
        print('saving policy at epoch {0}'.format(epoch))
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        torch.save(policy.state_dict(), os.path.join(outputdir, 'model_free_simple.' + outputmodelname))
        #Eval the new policy
        _, mean_reward = Eval()
        rewards.append(mean_reward)
        print("The mean reward is: " + str(mean_reward))
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
    outputmodelname="model.pth"
    #Define the environment
    num_clicks=100
    recom_number = 20
    args = get_args()
    bsize=args.batch_size
    embed_dim=args.embed_dim
    encod_dim=args.nhid
    init_embed=args.init_embed
    model_type=args.model
    embed_dim_policy = args.embed_dim_policy
    encod_dim_policy=args.nhid_policy
    seedkey=args.seed
    optim = args.optim
    #Load environment
    environment = "./model_output/environment.model.pth"
    env = Environment(bsize, embed_dim, encod_dim, num_clicks).to(device)
    env.load_state_dict(torch.load(environment))
    #Load initial policy
    policy_new = "./model_output/orig_policy.model.pth"
    policy = Policy(bsize, embed_dim_policy, encod_dim_policy, num_clicks-1, recom_number).to(device)
    #print(policy)
    policy.load_state_dict(torch.load(policy_new))
    torch.save(policy.state_dict(), 'model_free_simple.policy.pth')
    
    #Training for model-free reinforcement learning
    max_epoch = 200
    max_length = 5
    capacity = 10000
    origin_reward, _ = Eval()
    _, rewards, max_reward = train(optim, max_epoch, policy, env, num_clicks, recom_number, max_length, origin_reward, capacity)        
    #Plot rewards
    save_plot(max_epoch, 1, rewards, 'rewards_test.png')
    #Write rewards
    f=open('rewards_model_free.txt','ab+')
    np.savetxt(f,rewards)
    f.close()
