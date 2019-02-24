import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from evaluate import Evaluate
from replay import ReplayMemory
import torch.nn.functional as F

def Train_epoch(env, usr_fea_vec, model, target_model, optimizer, cuda):
    #Generate samples
    memory = ReplayMemory(capacity=20, batch_size=5)
    memory.sample_train(env, model, target_model, usr_fea_vec)
    values, rewards, next_values = memory.value, np.array(memory.reward), memory.next_value
    monte_values = np.array(memory.monte_value)
    log_value = memory.log_prob_value
    #Compute losses
    value_losses = []
    policy_losses = []
    if cuda:
        rewards = torch.FloatTensor(rewards).cuda()
        monte_values = torch.FloatTensor(monte_values).cuda()
    else:
        rewards = torch.FloatTensor(rewards)
        monte_values = torch.FloatTensor(monte_values)
    rewards = Variable(rewards)
    monte_values = Variable(monte_values)
    '''
    for (log_prob, value), r in zip(saved_actions, values):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.FloatTensor([r])))
    '''
    num = 0
    for i in range(len(rewards)):
        for j in range(len(rewards[i])):
            r1 = rewards[i][j] + env.gamma*next_values[i][j]
            r2 = monte_values[i][j]
            #r = 0.5*(r1+r2)
            r = r1
            policy_losses.append(-log_value[i][j] * rewards[i][j])
            value_losses.append(F.smooth_l1_loss(values[i][j], r))
            num += 1
    '''
    optimizer.zero_grad()
    loss1 = torch.stack(policy_losses).sum()
    if cuda:
        loss1 = loss1.cuda()
    loss1.backward(retain_graph=True)
    optimizer.step()
    '''
    optimizer.zero_grad()
    loss2 = torch.stack(value_losses).sum()
    #loss = loss1 + loss2
    if cuda:
        loss2 = loss2.cuda()
    loss2.backward()
    optimizer.step()
    #policy_loss=torch.stack(policy_losses).sum().data.cpu().numpy()
    value_loss=torch.stack(value_losses).sum().data.cpu().numpy()#/num
    return value_loss
    
def Train(epoch_num, model, target_model, env, usr_fea_vec, baseline, optimizer, cuda=True):
    #losses = []
    #policy_losses = []
    printfreq = 100
    target_update_freq = 10
    model.train()
    value_losses = []
    for i in range(epoch_num):
        #print("Epoch " + str(i))
        #Periodically update the target model
        if i % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
        value_loss=Train_epoch(env, usr_fea_vec, model, target_model, optimizer, cuda)
        value_losses.append(value_loss)
        if i % printfreq == 0:
            print("Epoch " + str(i))
            Evaluate(env, usr_fea_vec, model, baseline, False, cuda)
    save_plot(epoch_num, value_losses, "DQN.png")
    
def save_plot(epoch_num, value_losses, filepath):
    """Generate and save the plot"""
    fig, ax = plt.subplots()
    ax.plot(range(epoch_num), value_losses)
    fig.savefig(filepath)
    plt.close(fig)  # close the figure