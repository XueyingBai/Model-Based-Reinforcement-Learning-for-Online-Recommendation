import numpy as np
from rl_model import DQN
from usr_behavior_model import pp_net
from env_simu import Environment
from MDP_solver import LP, PolicyIteration 
import torch.optim as optim
import torch
from train import Train
from evaluate import Evaluate
from util import getargs

if __name__ == "__main__":
    args = get_args()
    global device, BATCH_SIZE
    BATCH_SIZE = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Seed
    seedkey=args.seed
    np.random.seed(seedkey)
    torch.manual_seed(seedkey)
    torch.cuda.manual_seed(seedkey)
    '''
    window=args.window
    '''
    #window = [0.1, 0.3, 0.5, 0.6, 0.8]
    pre_window=args.pre_window
    #Data
    samples, feature_vec, numlabel=ReadSeq(args.data,args.feature_vector)#,"itemIDTransform_session")
    #trainSample, validSample, testSample=sampleSplit(0.7,0.1,samples, pre_window)
    trainSample, testSample=sampleSplit(0.7,0.1,samples, pre_window)
    print('Train sample : {0}'.format(trainSample.length()))
    #print('Valid sample : {0}'.format(validSample.length()))
    print('Test sample : {0}'.format(testSample.length()))
    
    embed_dim=args.embed_dim
    
    encod_dim=args.nhid
    init_embed=args.init_embed
    model_type=args.model
    
    #Model
    pp_net = torch.load('../TimeSeries/model_output/model_cell.sequential.pickle')
    print(pp_net)
    pp_net = pp_net.to(device)
    param_dict = count_parameters(pp_net)
    print('number of trainable parameters = ', np.sum(list(param_dict.values())))
    '''
    #Train
    optims="Adam, lr=0.01"
    #loss    
    #optimizer
    optim_fn, optim_params=get_optimizer(optims)
    optimizer = optim_fn(filter(lambda p: p.requires_grad, pp_net.parameters()), **optim_params)
    '''
    global policy_net, target_net
    policy_net = DQN(encod_dim, numlabel).to(device)
    target_net = DQN(encod_dim, numlabel).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    global memory, steps_done
    #optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = ReplayMemory(10000)

    steps_done = 0   
    train(pp_net, trainSample, testSample)
    evaluate(pp_net, testSample)