from util import get_args, get_optimizer
import os
from data import ReadSeq_Gen, ReadSeq_Dis, sampleSplit
import numpy as np
import torch
from generator import Generator
from discriminator import Discriminator
import torch.nn as nn
from train import train_nll, train_gen_pg
from eval import evaluate
from helper import gen_fake, split_index, write_seq
import shutil

def adj_optim(optims, optimizer, minlr, lrshrink, stop_training, adam_stop):
    if 'sgd' in optims:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / lrshrink
        print('Shrinking lr by : {0}. New lr = {1}'.format(lrshrink, optimizer.param_groups[0]['lr']))
        if optimizer.param_groups[0]['lr'] < minlr:
            stop_training = True
    if 'adam' in optims:
        # early stopping (at 2nd decrease in accuracy)
        stop_training = adam_stop
        adam_stop = True 
    return stop_training, adam_stop
    
def train(optims, model_type, model, bsize, embed_dim, trainSample, validSample, testSample):
    outputdir="model_output"
    outputmodelname="model.pickle"
    lrshrink=5
    minlr=1e-5
    
    #weight = torch.FloatTensor(numlabel).fill_(1)
    loss_fn = nn.NLLLoss()
    loss_fn.size_average = False
    loss_fn.to(device)
    optim_fn, optim_params=get_optimizer(optims)
    optimizer = optim_fn(filter(lambda p: p.requires_grad, model.parameters()), **optim_params)
    
    n_epochs=200
    val_acc_best = -1e10  
    adam_stop = False
    stop_training = False
    epoch = 1
    eval_type = 'valid'
    while not stop_training and epoch <= n_epochs:
        train_acc, train_map = train_nll(model_type, model, epoch, trainSample, optimizer, bsize, embed_dim, loss_fn, device)        
        eval_acc, eval_map = evaluate(model_type, model, epoch, bsize, validSample, testSample, device, eval_type)         # save model
        if eval_type == 'valid' and epoch <= n_epochs:
            if eval_acc > val_acc_best:
                print('saving model at epoch {0}'.format(epoch))
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                if model_type == 'generator':
                    torch.save(model, os.path.join(outputdir, 'seqGan_gen2.' + outputmodelname))
                else:
                    torch.save(model, os.path.join(outputdir, 'seqGan_dis2.' + outputmodelname))
                val_acc_best = eval_acc
            else:
                stop_training, adam_stop = adj_optim(optims, optimizer, minlr, lrshrink, stop_training, adam_stop)
        epoch += 1
    return val_acc_best 

def advtrain(optims_gen, optims_dis, generator, discriminator, bsize, embed_dim, trainSample, validSample, testSample, val_acc_best):
    outputdir="model_output"
    outputmodelname="model.pickle"
    lrshrink=5
    minlr=1e-5   
     
    #Define the optimizer
    optim_fn_gen, optim_params_gen=get_optimizer(optims_gen)
    optimizer_gen = optim_fn_gen(filter(lambda p: p.requires_grad, generator.parameters()), **optim_params_gen)
    optim_fn_dis, optim_params_dis=get_optimizer(optims_gen)
    optimizer_dis = optim_fn_dis(filter(lambda p: p.requires_grad, discriminator.parameters()), **optim_params_dis)  
    '''
    adam_stop_gen = False
    stop_training_gen = False
    adam_stop_dis = False
    stop_training_dis = False
     
    lr_gen = optim_params_gen['lr'] if 'sgd' in optims_gen else None
    lr_dis = optim_params_dis['lr'] if 'sgd' in optims_dis else None
    '''
    n_epochs=200
    epoch = 1
    eval_type = 'valid'
    while epoch <= n_epochs:
        #Generator training       
        print('Generator Training')
        # Select subset of trainSample
        subnum = 8000
        _ = train_gen_pg(generator, discriminator, epoch, trainSample, subnum, optimizer_gen, bsize, embed_dim, 10, 10, device)
        print('Discriminator Training')        
        #Discriminator trainging
        shutil.copy('click_gen_real.txt', 'click_gen.txt')
        shutil.copy('tar_gen_real.txt', 'tar_gen.txt')
        _, _ = gen_fake(generator, trainSample, bsize, embed_dim, device, 5)
         
        clicklist = ReadSeq_Dis('click_gen.txt', 'tar_gen.txt')
        trainindex_dis, validindex_dis, testindex_dis = split_index(0.7, 0.1, len(clicklist))
        trainSample_dis, validSample_dis, testSample_dis=sampleSplit(trainindex_dis, validindex_dis, testindex_dis, clicklist, 'dis')
        _ = train(optims_dis, 'discriminator', discriminator, bsize, embed_dim, trainSample_dis, validSample_dis, testSample_dis)
        eval_acc, eval_map = evaluate('generator', generator, epoch, bsize, validSample, testSample, device, eval_type)         # save model
        if eval_type == 'valid' and epoch <= n_epochs:
            if eval_acc > val_acc_best:
                print('saving model at epoch {0}'.format(epoch))
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                torch.save(generator, os.path.join(outputdir, 'seqGan_gen2.' + outputmodelname))
                torch.save(discriminator, os.path.join(outputdir, 'seqGan_dis2.' + outputmodelname))
                val_acc_best = eval_acc
            '''
            else:
                _, _ = adj_optim(optims, optimizer, lrshrink, adam_stop)
            '''
        epoch += 1
    return val_acc_best
        
if __name__ == '__main__':
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.remove('click_gen_real.txt')
    os.remove('tar_gen_real.txt')
    #Parameters
    args = get_args()
    bsize=args.batch_size
    embed_dim=args.embed_dim
    encod_dim=args.nhid
    init_embed=args.init_embed
    model_type=args.model
    seedkey=args.seed
    optims=args.optim
    #Seed    
    np.random.seed(seedkey)
    torch.manual_seed(seedkey)
    torch.cuda.manual_seed(seedkey)
    #Load Sequence Data
    Seqlist, numlabel=ReadSeq_Gen(args.click,args.purchase)
    trainindex, validindex, testindex = split_index(0.7, 0.1, len(Seqlist)) 
    #Write training sequences for discriminator
    trainlist = []
    for i in trainindex:
        trainlist.append(Seqlist[i].getClick())
    write_seq(trainlist, 'click_gen_real.txt', 'tar_gen_real.txt', real = True)
    ''' 
    #Pretrain generator
    generator = Generator(bsize, embed_dim, encod_dim, numlabel).to(device)
    print(generator)
    trainSample, validSample, testSample=sampleSplit(trainindex, validindex, testindex, Seqlist)
    print('Train sample : {0}'.format(trainSample.length()))
    print('Valid sample : {0}'.format(validSample.length()))
    print('Test sample : {0}'.format(testSample.length()))
    val_acc_best = train(optims, 'generator', generator, bsize, embed_dim, trainSample, validSample, testSample)
    
    generator = torch.load('model_output/seqGan_gen.model.pickle')
    #Generate fake sequences
    trainSample, validSample, testSample=sampleSplit(trainindex, validindex, testindex, Seqlist, 'adv')
    print('Train sample : {0}'.format(trainSample.length()))
    print('Valid sample : {0}'.format(validSample.length()))
    print('Test sample : {0}'.format(testSample.length()))
    _, _ = gen_fake(generator, trainSample, bsize, embed_dim, device, 5)
    
    #Pretrain discriminator
    discriminator = Discriminator(bsize, embed_dim, encod_dim, numlabel, 2).to(device)
    print(discriminator)
    clicklist = ReadSeq_Dis('click_gen.txt', 'tar_gen.txt')
    trainindex_dis, validindex_dis, testindex_dis = split_index(0.7, 0.1, len(clicklist))
    trainSample, validSample, testSample=sampleSplit(trainindex_dis, validindex_dis, testindex_dis, clicklist, 'dis')
    print('Train sample : {0}'.format(trainSample.length()))
    print('Valid sample : {0}'.format(validSample.length()))
    print('Test sample : {0}'.format(testSample.length()))
    discriminator = train(optims, 'discriminator', discriminator, bsize, embed_dim, trainSample, validSample, testSample)
    ''' 
    #Adversarial training
    optims_gen = 'adam'
    #optims_dis = 'adagrad'
    #optims_gen = 'sgd,lr=0.2'
    optims_dis = 'adam'
    generator = torch.load('model_output/seqGan_gen.model.pickle')
    discriminator = torch.load('model_output/seqGan_dis.model.pickle')
    val_acc_best = 30.81
    trainSample, validSample, testSample=sampleSplit(trainindex, validindex, testindex, Seqlist, 'adv')
    _ = advtrain(optims_gen, optims_dis, generator, discriminator, bsize, embed_dim, trainSample, validSample, testSample, val_acc_best)
   
    
    
    
    
    
    