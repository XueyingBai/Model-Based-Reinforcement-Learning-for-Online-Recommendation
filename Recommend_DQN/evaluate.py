import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from replay import ReplayMemory

def Evaluate(env, usr_fea_vec, model, baseline, final=False, cuda=True):
    model.eval()
    if not final:
        batch_size=6
        capacity=20
    else:
        batch_size=60
        capacity=200
    memory = ReplayMemory(capacity, batch_size)
    value_loss = memory.sample_eval(env, model, baseline, usr_fea_vec)
    print('epoch_eval: value_loss: {0}'.format(value_loss))
    #if final:
    #state_value = []
    #state_action = []
    for i in range(len(env.S)):
        state = torch.from_numpy(env.S[i]).float()
        if cuda:
            state = state.cuda()        
        state = Variable(state, volatile=True)
        if final:
            print(model(state))
        
        state_value = model(state).max(1)[0].data.cpu().numpy()
        state_action = model(state).max(1)[1].data.cpu().numpy()
        '''
        state_action, state_value = model(state)
        state_action = state_action.max(1)[1].data.cpu().numpy()
        state_value = state_value.max(1)[0].data.cpu().numpy()
        '''
        print('state: {0}, next_action: {1}, value: {2}'.format(i, state_action, state_value))
