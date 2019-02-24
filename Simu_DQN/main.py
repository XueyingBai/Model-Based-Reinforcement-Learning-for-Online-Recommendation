import numpy as np
from model import DQN, ActCritic
from env_simu import Environment
from MDP_solver import LP, PolicyIteration 
import torch.optim as optim
import torch
from train import Train
from evaluate import Evaluate

if __name__ == '__main__':
    #Define the environment
    num_states=10
    fea_states=11
    num_actions=5
    fea_actions=6
    usr_num=20
    usr_fea=5
    env, usr_fea_vec = Environment(num_states, fea_states, fea_actions, num_actions, usr_num, usr_fea)
    #print(env.T)
    #Ground Truth policy
    #LP solver
    print("LP solver of the MDP")
    pi = LP(env.S, env.T, env.A, env.w, usr_fea_vec[0])
    pi.run()
    print(pi.policy)
    print(pi.V)
    print("Policy Iteration Solver")
    pi = PolicyIteration(env.S, env.T, env.A, env.w, usr_fea_vec[0], eval_type=1)
    pi.run()
    print("Policy is:")
    print(pi.policy)
    print("Value is:")
    print(pi.V)
    
    #Model setting
    cuda = True
    torch.cuda.set_device(1)
    model = DQN(fea_states, num_actions)
    #model = ActCritic(fea_states, num_actions)
    #model_target = ActCritic(fea_states, num_actions)
    model_target = DQN(fea_states, num_actions)
    model_target.load_state_dict(model.state_dict())
    model_target.eval()
    print(model)
    epoch_num = 100000
    if cuda:
        model = model.cuda()
        model_target = model_target.cuda()
    #optimizer = optim.RMSprop(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    Train(epoch_num, model, model_target, env, usr_fea_vec, pi.policy, optimizer, cuda)
    print('Testing')
    Evaluate(env, usr_fea_vec, model, pi.policy, True, cuda)