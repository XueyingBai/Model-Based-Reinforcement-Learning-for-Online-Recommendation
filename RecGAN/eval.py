import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from helper import getBatch_pred, getBatch_dis
from torch import optim
import numpy as np

def evaluate_interaction(model, epoch, batch_size, recom_length, validSample, testSample, loss_fn_target, loss_fn_reward, device, eval_type='valid', final_eval=False):
    correct = 0.
    correct_reward = 0.
    mapeach = 0.
    all_costs = []
        
    if eval_type == 'valid':
        sample =  validSample
    else: 
        sample = testSample
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    for i in range(0, sample.length(), batch_size):
        # prepare batch
        embed_batch, length, tgt_batch, reward_batch, action_batch = getBatch_pred(i, i + batch_size, sample, None, recom_length)
        embed_batch,tgt_batch, reward_batch, action_batch = Variable(embed_batch.cuda()), Variable(tgt_batch.cuda()), Variable(reward_batch.cuda()), Variable(action_batch.cuda())
        k = embed_batch.size(0)  
        # model forward
        generator, agent = model
        generator.eval()
        agent.eval()
        enc_out, h = generator((embed_batch, length))
        _, action, _ = agent((embed_batch, length), True)        
        output = generator.next_click(enc_out[:,-1,:], action, len(embed_batch))
        reward, reward_logit = generator.get_reward(tgt_batch.view(-1,1), enc_out[:,-1,:].unsqueeze(0))
        pred_reward = torch.round(reward.data)
        correct_reward += pred_reward.long().eq(reward_batch.data.long()).cpu().sum().numpy()  
           
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().numpy().sum()
        all_prob = output.data.cpu().numpy()
        for i in range(len(output)):
            mapeach += 1/int((np.argwhere(np.argsort(-all_prob[i])==tgt_batch.data.long().cpu().numpy()[i])[0]+1))
        # loss
        with torch.no_grad():
            loss_pred = loss_fn_target(output, tgt_batch)
            loss_reward = loss_fn_reward(reward_logit, reward_batch)
            loss = loss_pred + loss_reward
            all_costs.append(loss.data.cpu().numpy())  
        
    eval_acc  = np.round(100 * correct / sample.length(),2)
    eval_map=np.round(100 * mapeach / sample.length(),2) 
    eval_acc_reward = np.round(100 * correct_reward / sample.length(),2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}, map {0} : {2}, accuracy reward {0} : {3}'.format(eval_type, eval_acc, eval_map, eval_acc_reward))
    else:
        print('togrep : results : epoch {0} ; mean accuracy pred {1} : {2}, map pred {1} : {3}; mean accuracy reward {1} : {4}'.format(epoch, eval_type, eval_acc, eval_map, eval_acc_reward))
    return eval_acc, eval_map, eval_acc_reward, np.mean(all_costs)
    
def evaluate_agent(agent, epoch, batch_size, recom_length, validSample, testSample, device, eval_type='valid', final_eval=False):
    correct = 0.
    correctk = 0.
    
    if eval_type == 'valid':
        sample =  validSample
    else: 
        sample = testSample
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    for i in range(0, sample.length(), batch_size):
        # prepare batch
        embed_batch,length,tgt_batch, reward_batch, action_batch = getBatch_pred(i, i + batch_size, sample, None, recom_length)
        embed_batch,tgt_batch, action_batch, reward_batch = Variable(embed_batch.cuda()), Variable(tgt_batch.cuda()), Variable(action_batch.cuda()), Variable(reward_batch.cuda())
        k = embed_batch.size(0)  
        # model(agent) forward        
        agent.eval()
        probs, _, _ = agent((embed_batch, length), True)
        # mask to get reranking for offline actions
        mask = torch.zeros(k, probs.size(1)).cuda()
        mask.scatter_(1, action_batch, 1.)
        outputk = probs * mask 
        
        '''
        #print(output.data.long())  
        correct += output[:, 0].data.long().eq(tgt_batch.data.long()).cpu().numpy().sum()
        for i in range(len(output)):
            if tgt_batch[i].data.cpu().numpy() in output[i, :9].data.cpu().numpy():
                correctk += 1
        '''
        output_click = outputk.data.max(1)[1]
        correct += output_click.data.long().eq(tgt_batch.data.long()).cpu().numpy().sum()
        all_prob_output = outputk.data.cpu().numpy()
        
        for i in range(len(all_prob_output)):
            pos = int(np.argwhere(np.argsort(-all_prob_output[i])==tgt_batch.data.long().cpu().numpy()[i])[0]+1)
            # p@k
            if pos <= 10:
                correctk += 1                   
    #print()            
    eval_acc  = np.round(100 * correct / sample.length(),2)
    eval_prek = np.round(100*correctk / sample.length(),2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}, precision@k {0} : {2}'.format(eval_type, eval_acc, eval_prek))
    else:
        print('togrep : results : epoch {0} ; accuracy {1} : {2}, precision@10 {1} : {3}'.format(epoch, eval_type, eval_acc, eval_prek))
    return eval_acc, eval_prek 
    
def evaluate_user(generator, epoch, batch_size, recom_length, validSample, testSample, loss_fn_target, loss_fn_reward, device, eval_type='valid', model_type='recommend', final_eval=False):
    correct = 0.
    correctk = 0.
    correct_reward = 0.
    all_costs = []
    
    if eval_type == 'valid':
        sample =  validSample
    else: 
        sample = testSample
        print('\nVALIDATION : Epoch {0}'.format(epoch))
        
    with torch.no_grad():
        loss_fn_target = nn.CrossEntropyLoss()
        loss_fn_reward = nn.BCEWithLogitsLoss()
        loss_fn_target.size_average = True
        loss_fn_target.to(device)
        loss_fn_reward.size_average = True
        loss_fn_reward.to(device)
        
    for i in range(0, sample.length(), batch_size):
        # prepare batch
        embed_batch,length,tgt_batch, reward_batch, action_batch = getBatch_pred(i, i + batch_size, sample, None, recom_length)
        embed_batch,tgt_batch, reward_batch, action_batch = Variable(embed_batch.cuda()), Variable(tgt_batch.cuda()), Variable(reward_batch.cuda()), Variable(action_batch.cuda())
        k = embed_batch.size(0)  
        # model(agent) forward        
        generator.eval()
        enc_out, h = generator((embed_batch, length))
        if model_type == 'recommend':
            output = generator.next_click(enc_out[:,-1,:], action_batch, len(embed_batch))
        else:
            output = generator.next_simple(enc_out[:,-1,:])
        output_click = output.data.max(1)[1]
        correct += output_click.data.long().eq(tgt_batch.data.long()).cpu().numpy().sum()
        all_prob_output = output.data.cpu().numpy()
        
        reward, reward_logit = generator.get_reward(tgt_batch.view(-1,1), enc_out[:,-1,:].unsqueeze(0))
        pred_reward = torch.round(reward)
        correct_reward += pred_reward.long().eq(reward_batch.data.long()).cpu().sum().numpy()  
        
        for i in range(len(all_prob_output)):
            pos = int(np.argwhere(np.argsort(-all_prob_output[i])==tgt_batch.data.long().cpu().numpy()[i])[0]+1)
            # p@k
            if pos <= 10:
                correctk += 1                   
        # loss
        with torch.no_grad():
            loss_pred = loss_fn_target(output, tgt_batch)
            loss_reward = loss_fn_reward(reward_logit, reward_batch)
            loss = loss_pred + loss_reward
            all_costs.append(loss.data.cpu().numpy())  
        
    eval_acc  = np.round(100 * correct / sample.length(),2)
    eval_prek = np.round(100*correctk / sample.length(),2)
    eval_acc_rewd = np.round(100*correct_reward / sample.length(),2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}, precision@k {0} : {2}'.format(eval_type, eval_acc, eval_prek))
    else:
        print('togrep : results : epoch {0} ; accuracy {1} : {2}, precision@10 {1} : {3}, reward_accuracy {1} {4}'.format(epoch, eval_type, eval_acc, eval_prek, eval_acc_rewd))
    return eval_acc, eval_prek, eval_acc_rewd, np.mean(all_costs) 
    
def evaluate_discriminator(discriminator, epoch, batch_size, recom_length, validSample, testSample, device, eval_type='valid', final_eval=False):
    correct = 0.
    correct_reward = 0.
    mapeach = 0.
    discriminator.eval()
    if eval_type == 'valid':
        sample =  validSample
    else: 
        sample = testSample
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    for i in range(0, sample.length(), batch_size):
        # prepare batch
        embed_batch, length, tgt_batch, reward_batch, action_batch = getBatch_dis(i, i + batch_size, sample, None, recom_length)
        reward_batch=reward_batch.type(torch.FloatTensor)
        embed_batch,tgt_batch, reward_batch, action_batch = Variable(embed_batch.cuda()), Variable(tgt_batch.cuda()), Variable(reward_batch.cuda()), Variable(action_batch.cuda())
        k = embed_batch.size(0) 
         
        output = discriminator((embed_batch, length), reward_batch, action_batch)       
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().numpy().sum()
        all_prob = output.data.cpu().numpy()
        for i in range(len(output)):
            mapeach += 1/int((np.argwhere(np.argsort(-all_prob[i])==tgt_batch.data.long().cpu().numpy()[i])[0]+1))
                
    eval_acc  = np.round(100 * correct / sample.length(),2)
    eval_map=np.round(100 * mapeach / sample.length(),2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}, map {0} : {2}'.format(eval_type, eval_acc, eval_map))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} : {2}, map {1} : {3}'.format(epoch, eval_type, eval_acc, eval_map))
    return eval_acc, eval_map 
