import numpy as np
from policy_origin import Policy
from MDP import Environment
from MDP_solver import LP, PolicyIteration 
import torch.optim as optim
import torch
from replay import ReplayMemory
import os
import pickle
def save(parameter, filename):
    with open(filename, 'wb') as f:
        pickle.dump(parameter, f)
        
if __name__ == '__main__':
    #Define the environment
    num_states=20
    fea_states=11
    num_actions=40
    fea_actions=6
    usr_num=20
    usr_fea=5
    env, usr_fea_vec, end_state = Environment(num_states, fea_states, fea_actions, num_actions, usr_num, usr_fea)
    #Save the environment
    save(env, 'environment.pickle')
    save(usr_fea_vec, 'usr_fea_vec.pickle')
    save(end_state, 'end_state.pickle')
    print(end_state)
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
    #torch.cuda.set_device(1)
    #Generate the starting states
    model_policy = Policy(fea_states, num_actions).cuda()
    print(model_policy)
    #Save the model    
    model_policy.init_params()
    outputdir="model_output"
    outputmodelname="model.pickle"
    env_policy = torch.save(model_policy, os.path.join(outputdir, 'env_policy.' + outputmodelname))    
    #Generate the start states
    capacity = 10000
    max_length = 10
    batch_size = 128
    file_action = 'gen_action.txt'
    file_reward = 'gen_reward.txt'
    if os.path.isfile(file_action):
        os.remove(file_action)
    if os.path.isfile(file_reward):
        os.remove(file_reward)
    Replay = ReplayMemory(env, model_policy, capacity, max_length, num_states, num_actions, end_state)
    #Save the start states
    save(Replay.states, 'start_states.pickle')
    Replay.gen_sample(usr_fea_vec[0], batch_size)
    Replay.write_sample(file_action, file_reward, num_actions, add_end=False)  
      
        
    