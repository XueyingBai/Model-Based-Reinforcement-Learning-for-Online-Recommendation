from util import get_args, get_optimizer
import os
from data import ReadSeq, sampleSplit
import numpy as np
import torch
from generator import Generator
from discriminator import Discriminator
from agent import Agent 
import torch.nn as nn
from train import train_pred_each, train_dis_each, train_gen_pg_each
from eval import evaluate_interaction, evaluate_discriminator, evaluate_agent, evaluate_user
from helper import gen_fake, split_index, write_seq, write_seq_reward, write_seq_action, plot_data_dist, save_plot
import shutil
import sys
import random
sys.path.append('../simulation_task1')
from evaluate import Eval
import subprocess

#Adjust optimizers
def adj_optim(optims, optimizer, minlr, lrshrink, stop_training, times_no_improvement):
    if 'sgd' in optims:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / lrshrink
        print('Shrinking lr by : {0}. New lr = {1}'.format(lrshrink, optimizer.param_groups[0]['lr']))
        if optimizer.param_groups[0]['lr'] < minlr:
            stop_training = True
    if 'adam' in optims:
        # early stopping (at 2nd decrease in accuracy)
        if times_no_improvement >= 2:
            stop_training = True
    return stop_training

#Training for the predictions of generators, NLL loss(with two optimizers) 
def train_user_pred(optims, generator, bsize, embed_dim, recom_length, trainSample, validSample, testSample, mode = 'generator with rec', inner_val_acc_best = None, inner_val_preck_best = None, inner_val_rewd_best = None, inner_loss_best = None, only_rewards = False, n_epochs=10):
    outputdir="model_output"
    outputmodelname="simu.model.pth"
    lrshrink=5
    minlr=1e-5
    generator_only = True
    action_given = True
    #Define the optimizers
    #loss_fn_target = nn.CrossEntropyLoss(weight)
    loss_fn_target = nn.CrossEntropyLoss()
    #loss_fn_reward = nn.MSELoss()
    loss_fn_reward = nn.BCEWithLogitsLoss()
    loss_fn_target.size_average = True
    loss_fn_target.to(device)
    loss_fn_reward.size_average = True
    loss_fn_reward.to(device)
    
    optim_fn, optim_params=get_optimizer(optims)
    if mode == 'generator':
        params = list(generator.parameters())
        action_given = False
    elif mode == 'generator with rec':
        params = list(generator.parameters())
        action_given = True
    else:
        print("No such mode! Select from generator/generator with rec!")
        
    optimizer = optim_fn(filter(lambda p: p.requires_grad, params), **optim_params)
    
    #n_epochs=10
    if inner_val_acc_best == None:
        inner_val_acc_best = -1e10 
        inner_val_preck_best = -1e10 
        inner_val_rewd_best = -1e10 
        inner_loss_best = 1e10
    stop_training = False
    times_no_improvement = 0
    epoch = 1
    eval_type = 'valid'
    best_model = generator
    while not stop_training and epoch <= n_epochs:
        if not only_rewards:
            #Train click
            #print("Clicks!")
            train_acc, train_preck, _ = train_pred_each(generator, epoch, trainSample, optimizer, bsize, embed_dim, recom_length, loss_fn_target, loss_fn_reward, device, generator_only, action_given, False)
        #Evaluate without EOS
        print("User model evaluation!")
        eval_acc, eval_preck, eval_rewd, eval_loss = evaluate_user(generator, epoch, bsize, recom_length-1, validSample, testSample, loss_fn_target, loss_fn_reward, device, eval_type)
        # save model
        if eval_type == 'valid' and epoch <= n_epochs:
            if eval_acc > inner_val_acc_best or eval_preck > inner_val_preck_best:
            #if inner_loss_best >= eval_loss:
                best_model = generator
                print('saving model at epoch {0}'.format(epoch))
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                torch.save(generator.state_dict(), os.path.join(outputdir, 'irecGan_gen3.' + outputmodelname))
                inner_val_acc_best = eval_acc
                inner_val_preck_best = eval_preck
                inner_val_rewd_best = eval_rewd
                inner_loss_best = eval_loss
                times_no_improvement = 0
            else:
                times_no_improvement += 1
                stop_training = adj_optim(optims, optimizer, minlr, lrshrink, stop_training, times_no_improvement)
        epoch += 1
    return best_model, inner_val_acc_best, inner_val_preck_best, inner_val_rewd_best, inner_loss_best 

#Training for the predictions of discriminators, NLL loss(with two optimizers)     
def train_dis(optims, model, bsize, embed_dim, recom_length, trainSample, validSample, testSample):
    outputdir="model_output"
    outputmodelname="simu.model.pth"
    lrshrink=5
    minlr=1e-5
    
    #weight = torch.FloatTensor(numlabel).fill_(1)
    loss_fn = nn.NLLLoss()
    loss_fn.size_average = True
    loss_fn.to(device)
    optim_fn, optim_params=get_optimizer(optims)
    optimizer = optim_fn(filter(lambda p: p.requires_grad, model.parameters()), **optim_params)
    
    n_epochs=5
    inner_val_acc_best = -1e10 
    inner_val_map_best = -1e10  
    stop_training = False
    epoch = 1
    eval_type = 'valid'
    best_model = model
    while not stop_training and epoch <= n_epochs:
        train_acc, train_map = train_dis_each(model, epoch, trainSample, optimizer, bsize, embed_dim, recom_length, loss_fn, device)        
        # Evaluate no eos
        eval_acc, eval_map = evaluate_discriminator(model, epoch, bsize, recom_length, validSample, testSample, device, eval_type)         # save model
        if eval_type == 'valid' and epoch <= n_epochs:
            if eval_acc > inner_val_acc_best or eval_map > inner_val_map_best:
                best_model = model
                print('saving model at epoch {0}'.format(epoch))
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                torch.save(model.state_dict(), os.path.join(outputdir, 'irecGan_dis.' + outputmodelname))
                inner_val_acc_best = eval_acc
                inner_val_map_best = eval_map
                times_no_improvement = 0
            else:
                times_no_improvement += 1
                stop_training = adj_optim(optims, optimizer, minlr, lrshrink, stop_training, times_no_improvement)
        epoch += 1
    return best_model, inner_val_acc_best, inner_val_map_best 

#Adversarial training with policy gradient loss
def pgtrain(optims_gen, optims_dis, generator, agent, discriminator, bsize, embed_dim, trainSample, validSample, testSample, val_acc_best, val_preck_best, val_loss_best, action_num, max_length, recom_length, gen_ratio = 0.1, n_epochs = 5, write_item = 'click_gen.txt', write_target='tar_gen.txt', write_reward = 'reward_gen.txt', write_action = 'action_gen.txt', plot_fig = True, pretrain = False):
    outputdir="model_output"
    outputmodelname="simu.model.pth"
    lrshrink=5
    minlr=1e-5    
    
    #Evaluation loss functions
    loss_fn_target = nn.CrossEntropyLoss()
    loss_fn_reward = nn.BCEWithLogitsLoss()
    loss_fn_target.size_average = True
    loss_fn_target.to(device)
    loss_fn_reward.size_average = True
    loss_fn_reward.to(device)
    
    inner_val_preck_best = val_preck_best
    inner_val_acc_best = val_acc_best 
    inner_loss_best = val_loss_best
    epoch = 1
    eval_type = 'valid'
    g_step = 1
    d_step = 1
    evalacc_all = [val_acc_best]
    evalpreck_all = [val_preck_best]
    #Define the optimizer
    optim_fn_gen, optim_params_gen=get_optimizer(optims_gen)
    optim_fn_dis, optim_params_dis=get_optimizer(optims_dis)
    optimizer_dis = optim_fn_dis(filter(lambda p: p.requires_grad, discriminator.parameters()), **optim_params_dis) 
    params_agent = list(agent.parameters())
    params_usr = list(generator.parameters())                    
    optimizer_agent = optim_fn_gen(filter(lambda p: p.requires_grad, params_agent), **optim_params_gen)
    optimizer_usr = optim_fn_gen(filter(lambda p: p.requires_grad, params_usr), **optim_params_gen)
    while epoch <= n_epochs:  
        print('\nAdversarial Policy Gradient Training!')
        # Select subset of trainSample
        subnum = 8000
        for i in range(g_step):  
            print('G-step')
            if pretrain:
                print('For Pretraining')
                _ = train_gen_pg_each(generator, agent, discriminator, epoch, trainSample, trainSample.length(), optimizer_agent, optimizer_usr, bsize, embed_dim, recom_length, max_length, action_num, device, 0, pretrain)
            else:
                print('For Policy Gradient Update')
                #shuffle_index=np.random.permutation(origin.length()) 
                _ = train_gen_pg_each(generator, agent, discriminator, epoch, trainSample, subnum, optimizer_agent, optimizer_usr, bsize, embed_dim, recom_length, max_length, action_num, device, 0.1, pretrain)
                
        # save model 
        # Evaluate without eos, no eos input
        print("Agent evaluation!")   
        eval_acc, eval_preck = evaluate_agent(agent, epoch, bsize, recom_length, validSample, testSample, device, eval_type='valid') 
        print("User model evaluation!")
        _ = evaluate_user(generator, epoch, bsize, recom_length, validSample, testSample, loss_fn_target, loss_fn_reward, device, eval_type)
        print("Interaction evaluation!")
        _ = evaluate_interaction((generator, agent), epoch, bsize, recom_length, validSample, testSample, loss_fn_target, loss_fn_reward, device, eval_type) 
        
        evalacc_all.append(eval_acc)
        evalpreck_all.append(eval_preck)
        if eval_type == 'valid' and epoch <= n_epochs:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            torch.save(agent.state_dict(), os.path.join(outputdir, 'irecGan_agent3.' + outputmodelname))
            torch.save(generator.state_dict(), os.path.join(outputdir, 'irecGan_gen3.' + outputmodelname))
            
            inner_val_acc_best = eval_acc
            inner_val_preck_best = eval_preck
             
        if not pretrain:
            '''
            #Adjust the reward prediction
            print('Reward Adjust')
            trainSample_rewd, validSample_rewd, testSample_rewd=sampleSplit(trainindex, validindex, testindex, Seqlist, numlabel, recom_length)
            _ = train_user_pred(optims_dis, generator, bsize, embed_dim, recom_length + 1, trainSample_rewd, validSample_rewd, testSample_rewd, 'generator with rec', None, None, None, None, only_rewards = True, n_epochs=1)
            #Enable full model training
            for name, param in generator.named_parameters():
                if 'embedding' in name or 'encoder' or 'enc2out' in name:
                    param.requires_grad = True
            '''
            print('\nD-step')        
            #Discriminator trainging
            for i in range(d_step):
                shutil.copy('click_gen_real.txt', write_item)
                shutil.copy('reward_gen_real.txt', write_reward)
                shutil.copy('tar_gen_real.txt', write_target)
                shutil.copy('action_gen_real.txt', write_action)
                _, _, _, _ = gen_fake(generator, agent, trainSample, bsize, embed_dim, device, write_item, write_target, write_reward, write_action, action_num, max_length, recom_length)
                clicklist, _ = ReadSeq(write_item, write_reward, write_action, write_target)
                trainindex_dis, validindex_dis, testindex_dis = split_index(0.7, 0.1, len(clicklist), True) #Shuffle the index
                trainSample_dis, validSample_dis, testSample_dis=sampleSplit(trainindex_dis, validindex_dis, testindex_dis, clicklist, 2, recom_length, 'dis')

                discriminator, _, _ = train_dis(optims_dis, discriminator, bsize, embed_dim, recom_length, trainSample_dis, validSample_dis, testSample_dis)           
        epoch += 1
        
    if plot_fig == True:
        save_plot(n_epochs, 1, evalacc_all, 'pg_accuracy6.png')
        save_plot(n_epochs, 1, evalpreck_all, 'pg_map6.png')
    return inner_val_acc_best, inner_val_preck_best
        
if __name__ == '__main__':
    ''' 
    if os.path.isfile('myprog.log'):
        os.remove('myprog.log')
    log = open("myprog.log", "a")
    sys.stdout = log
    '''  
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Parameters 
    args = get_args()
    bsize=args.batch_size
    embed_dim=args.embed_dim
    encod_dim=args.nhid
    embed_dim_policy = args.embed_dim_policy
    encod_dim_policy=args.nhid_policy
    init_embed=args.init_embed
    model_type=args.model
    seedkey=args.seed
    optims_nll=args.optim_nll
    optims_adv=args.optim_adv
    n_layers_usr=args.n_layers_usr
    n_layers_agnt=args.n_layers_agnt
    pad = args.pad
    max_length=5
    #Seed    
    np.random.seed(seedkey)
    torch.manual_seed(seedkey)
    torch.cuda.manual_seed(seedkey)
    
    #Routes
    pretrained_gen =  'model_output/irecGan_gen3.simu.model.pth'
    pretrained_dis =  'model_output/irecGan_dis.simu.model.pth'
    pretrained_agent = 'model_output/irecGan_agent3.simu.model.pth'
    final_dis = 'model_output/irecGan_dis1.model.simu.pth'
    #Define the writing process
    write_item = 'click_gen.txt'
    write_target = 'tar_gen.txt'
    write_reward = 'reward_gen.txt'
    write_action = 'action_gen.txt'

    global generator, agent, discriminator
    
    subprocess_cmd = "python ../simulation_task1/Generate_data.py --load --embed_dim 50 --nhid 20 --embed_dim_policy 20 --nhid_policy 20"
    subprocess_cmd = subprocess_cmd.split()
    
    rewards = []
    Epochs = 3
    global interact
    interact = True
    '''
    val_acc_best = None 
    val_map_best = None
    val_loss_best = None
    '''
    for e in range(Epochs):
        print("==============================================")
        print("Training on the epoch:" + str(e))
        print("==============================================")
        
        if os.path.isfile('click_gen_real.txt'):
            os.remove('click_gen_real.txt')
        if os.path.isfile('tar_gen_real.txt'):
            os.remove('tar_gen_real.txt')
        if os.path.isfile('reward_gen_real.txt'):
            os.remove('reward_gen_real.txt')
        if os.path.isfile('action_gen_real.txt'):
            os.remove('action_gen_real.txt')

        #Load Sequence Data
        Seqlist, itemlist=ReadSeq(args.click, args.reward, args.action)
        numlabel = len(itemlist) + 1 #add end
        print(numlabel)
        #print("The real time of items appear:")
        total = 0
        for i in itemlist:
            total += itemlist[i]
        #print(total/len(itemlist))
        #print(itemlist)
        plot_data_dist(itemlist, 'data.png')
        
        #Generate the training, validating and testing sequence ids
        global trainindex, validindex, testindex
        trainindex, validindex, testindex = split_index(0.8, 0.1, len(Seqlist))  
        print('Train seq : {0}'.format(len(trainindex)))
        print('Valid seq : {0}'.format(len(validindex)))
        print('Test seq : {0}'.format(len(testindex)))
        #Write real training sequences for discriminator
        trainclicklist = []
        trainrewardlist = []
        trainactionlist = []
        for i in trainindex:
            trainclicklist.append(Seqlist[i].click)
            trainrewardlist.append(Seqlist[i].reward)
            trainactionlist.append(Seqlist[i].action)        
        write_seq(trainclicklist, 'click_gen_real.txt', 'tar_gen_real.txt', 'dis', real = True)
        write_seq_reward(trainrewardlist, 'reward_gen_real.txt')
        write_seq_action(trainactionlist, 'action_gen_real.txt')

        #lower weight for the end label
        #global weight
        weight = torch.FloatTensor(numlabel).fill_(1) 
        #weight[-1] = 0.001
        weight = weight.cuda()
        #index = torch.tensor([0])
        #weight.index_fill_(0, index, 0.5)

        #Set recommendation length
        recom_length = 0
        for i in range(len(Seqlist)):
            for act in Seqlist[i].action:
                if len(act)>recom_length:
                    recom_length = len(act)
        print(recom_length)
        #Add EOS
        recom_length += 1
        
        if e == 0:
            generator = Generator(bsize, embed_dim, encod_dim, numlabel, n_layers_usr, model='LSTM').to(device)
            #Agent will not recommend EOS
            agent = Agent(bsize, embed_dim_policy, encod_dim_policy, numlabel-1, n_layers_agnt, recom_length-1, model=model_type).to(device) 
            discriminator = Discriminator(bsize, embed_dim, encod_dim, embed_dim_policy, encod_dim_policy, numlabel, recom_length-1, 2).to(device) 
           
            print("The generator is:")
            print(generator)
            print("The agent is:")
            print(agent)
            print("The discriminator is:")
            print(discriminator)
             
        #Loss for test
        with torch.no_grad():
            loss_fn_target = nn.CrossEntropyLoss()
            loss_fn_reward = nn.BCELoss()
            loss_fn_target.size_average = False
            loss_fn_target.to(device)
            loss_fn_reward.size_average = False
            loss_fn_reward.to(device)
         
        trainSample, validSample, testSample=sampleSplit(trainindex, validindex, testindex, Seqlist, numlabel, recom_length-1)#, warm_up = 0)
        
        #Pretrain generator only
        val_acc_best, val_preck_best, val_rewd_best, val_loss_best = None, None, None, None
         
        print('\n--------------------------------------------')
        print("Pretrain Generator with given recommendation")
        print('--------------------------------------------')
        generator, val_acc_best, val_preck_best, val_rewd_best, val_loss_best = train_user_pred(optims_nll, generator, bsize, embed_dim, recom_length, trainSample, validSample, testSample, 'generator with rec', val_acc_best, val_preck_best, val_rewd_best, val_loss_best, n_epochs=80) 
         
        #Test pretrained user model
        print("Testing")
        print("User model evaluation!")
        #generator.load_state_dict(torch.load(pretrained_gen))
        #Evaluate without EOS
        generator.load_state_dict(torch.load(pretrained_gen))
        eval_acc, eval_preck, eval_rewd, eval_loss = evaluate_user(generator, 101, bsize, recom_length-1, validSample, testSample, loss_fn_target, loss_fn_reward, device, 'test')
           
        print('\n--------------------------------------------')
        print('Pretrain by the adversarial training')
        print('---------------------------------------------') 
        generator.load_state_dict(torch.load(pretrained_gen))
        trainSample, validSample, testSample=sampleSplit(trainindex, validindex, testindex, Seqlist, numlabel, recom_length-1, 'adv')
         
        _ = pgtrain(optims_adv, optims_nll, generator, agent, discriminator, bsize, embed_dim, trainSample, validSample, testSample, None, None, None, numlabel, max_length, recom_length-1, gen_ratio = 1, n_epochs = 20, plot_fig = False, pretrain = True)
          
        #Test pretrained agent model 
        print("Testing")
        agent.load_state_dict(torch.load(pretrained_agent)) 
        print("Agent evaluation!")   
        eval_acc, eval_preck = evaluate_agent(agent, 101, bsize, recom_length-1, validSample, testSample, device, 'test')
        
        #Generate fake sequences, only use the training data
        generator.load_state_dict(torch.load(pretrained_gen))
        agent.load_state_dict(torch.load(pretrained_agent))
        trainSample, validSample, testSample=sampleSplit(trainindex, validindex, testindex, Seqlist, numlabel, recom_length-1, 'gen')
        print('Generate sample : {0}'.format(trainSample.length()))

        shutil.copy('click_gen_real.txt', write_item )
        shutil.copy('tar_gen_real.txt', write_target)
        shutil.copy('reward_gen_real.txt', write_reward)
        shutil.copy('action_gen_real.txt', write_action)
        _ = gen_fake(generator, agent, trainSample, bsize, embed_dim, device, write_item, write_target, write_reward, write_action, numlabel, max_length, recom_length-1) #No EOS
         
        #Pretrain discriminator
        print('\n--------------------------------------------')
        print("Pretrain the Discriminator")
        print('--------------------------------------------')
        dis_clicklist, _ = ReadSeq(write_item, write_reward, write_action, write_target)
        trainindex_dis, validindex_dis, testindex_dis = split_index(0.8, 0.1, len(dis_clicklist), True)
        trainSample, validSample, testSample=sampleSplit(trainindex_dis, validindex_dis, testindex_dis, dis_clicklist, 2, recom_length-1, 'dis')
        print('Train sample : {0}'.format(trainSample.length()))
        print('Valid sample : {0}'.format(validSample.length()))
        print('Test sample : {0}'.format(testSample.length()))
        weight = torch.FloatTensor(2).fill_(1)
        discriminator, _, _ = train_dis(optims_nll, discriminator, bsize, embed_dim, recom_length-1, trainSample, validSample, testSample)
        print("Testing")
        print("Discriminator evaluation!")   
        eval_acc_dis, eval_map_dis = evaluate_discriminator(discriminator, 101, bsize, recom_length-1, validSample, testSample, device, 'test') 
         
        #Adversarial training
        weight = torch.FloatTensor(2).fill_(1)
        print('\n--------------------------------------------')
        print("Adversarial Training")
        print('--------------------------------------------') 
        generator.load_state_dict(torch.load(pretrained_gen))
        discriminator.load_state_dict(torch.load(pretrained_dis))
        agent.load_state_dict(torch.load(pretrained_agent))
          
        trainSample, validSample, testSample=sampleSplit(trainindex, validindex, testindex, Seqlist, numlabel, recom_length-1, 'adv')#No eos
         
        _ = evaluate_agent(agent, 101, bsize, recom_length-1, validSample, testSample, device, 'test') 
        _ = pgtrain(optims_adv, optims_nll, generator, agent, discriminator, bsize, embed_dim, trainSample, validSample, testSample, eval_acc, eval_preck, None, numlabel, max_length, recom_length-1, n_epochs = 10)
         
        print("Testing")
        agent.load_state_dict(torch.load(pretrained_agent)) 
        print("Agent evaluation!")   
        _ = evaluate_agent(agent, 101, bsize, recom_length-1, validSample, testSample, device, 'test')
        print("User model evaluation!")
        #generator.load_state_dict(torch.load(pretrained_gen))
        #Evaluate without EOS
        generator.load_state_dict(torch.load(pretrained_gen))
        eval_acc, eval_preck, eval_rewd, eval_loss = evaluate_user(generator, 101, bsize, recom_length-1, validSample, testSample, loss_fn_target, loss_fn_reward, device, 'test')
        #Save the whole policy model
        torch.save(agent, 'model_output/agent.pickle')
         
        if interact: 
        #Generate new samples from the environment
            reward_orig, reward_optim = Eval('model_output/agent.pickle')
            if e == 0:
                rewards.append(reward_orig)
            rewards.append(reward_optim)
            '''  
            #Load the best model
            generator.load_state_dict(torch.load(pretrained_gen))
            discriminator.load_state_dict(torch.load(pretrained_dis))
            agent.load_state_dict(torch.load(pretrained_agent))
            '''
            #Generate new data
            subprocess.call(subprocess_cmd, shell=False)
    save_plot(Epochs, 1, rewards, 'all_rewards.png')
