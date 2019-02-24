import torch
from torch.autograd import Variable
import numpy as np
from evaluate import evaluate
from replay import getState_preference, getReward, select_click
import torch.nn.functional as F
from helper import Normalize

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0, 0
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    '''
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]) 
    '''   
    non_final_next_states = Variable(torch.stack(batch.next_state))
    state_batch = Variable(torch.stack(batch.state))
    action_batch = Variable(torch.stack(batch.action).squeeze(1))
    reward_batch = Variable(torch.stack(batch.reward).squeeze(1))
    next_preferences = Variable(torch.stack(batch.preference))
    a= policy_net(state_batch)
    #print(non_final_next_states.size())

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch) 
    
    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    next_state_values = target_net(non_final_next_states).max(1)[0].detach()
    next_state_values = next_state_values.view(-1,1)
    #next_choice_values = torch.mul(next_state_values, next_preferences)#.max(1)[0].detach()
    #next_choice_values = next_choice_values.max(1)[0].detach().view(-1,1)
    # Compute the expected Q values 
    expected_state_action_values = (next_state_values * 0.99) + reward_batch.unsqueeze(1)
     
    #Normalize two values
    state_action_values = Normalize(state_action_values) 
    expected_state_action_values = Normalize(expected_state_action_values)
     
    # Compute Huber loss
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)/reward_batch.size(0)
    loss = nn.MSELoss()
    output = loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    #loss.backward()
    output.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    #return output
    #print(output)
    return output, torch.sum(state_action_values-expected_state_action_values)

def train(ppnet, trainSample, validateSample, window = 0.5, num_episodes = 1000, k=20):
    TARGET_UPDATE = 10
    simu_no = 50
    MAX_SEQ_LEN = 100
    PRINT_FREQ = 100
    EVAL_FREQ = 100
    loss_all = [] 
    loss_real_all = []
    #value_loss = []
    reward_loss = []
    
    for i_episode in range(num_episodes):
        # Initialize the environment and state: select part of the sequence as the initial    
        seq_no = random.randrange(trainSample.length())
        length_batch_start, item_batch_start, time_batch_start, tgt_item_start, tgt_time_start, purchase = getSeq(seq_no, trainSample)
        loss = 0
        loss_real = 0
        for simu_no in range(simu_no):
            length_batch, item_batch, time_batch, tgt_item, tgt_time = length_batch_start, item_batch_start, time_batch_start, tgt_item_start, tgt_time_start
            state, preference = getState_preference(ppnet, length_batch, item_batch, time_batch)
            for t in range(MAX_SEQ_LEN):
                # Select and perform an action
                #recommendation = select_click(state, preference.size())
                #action = getClick(action, preference, k)
                action = select_click(state, preference, k)
                #Calculate the reward
                reward = getReward(action, purchase)
                reward = torch.FloatTensor(reward).to(device)
                # Observe new state           
                action_tensor = torch.from_numpy(np.array([action])).to(device).unsqueeze(1)
                item_batch = torch.cat([item_batch, action_tensor], 0) 
                length_batch = length_batch + 1
                next_state, next_preference = getState_preference(ppnet, length_batch, item_batch, time_batch)
    
                # Store the transition in memory
                # Convert to the tensor
                state_tensor = state
                next_state_tensor = next_state           
                reward_tensor = reward
                next_preference_tensor = next_preference
                
                memory.push(state_tensor, action_tensor, next_state_tensor, next_preference_tensor, reward_tensor)
                # Move to the next state
                state = next_state
                preference = next_preference
    
                # Perform one step of the optimization (on the target network)
                loss_step, loss_real_step = optimize_model()
                loss += loss_step 
                loss_real += loss_real_step
                # A sequence is done
                if t == MAX_SEQ_LEN-1:
                    if len(memory) >= BATCH_SIZE:
                        loss_all.append(loss.data.cpu().numpy()/(t+1)) 
                        loss_real_all.append(loss_real.data.cpu().numpy()/(t+1))     
                    else:
                        loss_all.append(0) 
                        loss_real_all.append(0)        
                    #break
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        # Print the Loss
        if i_episode % PRINT_FREQ == 0:
            print("Average training loss for episode {0}: {1}".format(i_episode, loss/(t+1)))
            
        if i_episode % EVAL_FREQ == 0:
            print("Evaluation!")
            reward_loss_each = evaluate(ppnet, validateSample)
            #value_loss.append(value_loss_each)
            reward_loss.append(reward_loss_each)
            
    print('Complete Training!')
    num_sample = len(loss_real_all)
    save_plot(num_sample, 1, loss_real_all, 'Walmart_DQN_trainrealvalue_diff_6.png')
    save_plot(num_sample, 1, loss_all, 'Walmart_DQN_trainvalue_diff_6.png')
    #save_plot(num_episodes, EVAL_FREQ, value_loss, 'Walmart_DQN_value_diff_6.png')
    save_plot(num_episodes, EVAL_FREQ, reward_loss, 'Walmart_DQN_reward_diff_6.png')