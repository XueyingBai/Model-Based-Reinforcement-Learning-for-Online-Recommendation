import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from helper import getBatch
from torch import optim
import numpy as np

def evaluate(model_type, model, epoch, batch_size, validSample, testSample, device, eval_type='valid', final_eval=False):
    model.eval()
    correct = 0.
    mapeach = 0
    
    if eval_type == 'valid':
        sample =  validSample
    else: 
        sample = testSample
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    for i in range(0, sample.length(), batch_size):
        # prepare batch
        embed_batch,length,tgt_batch, pur_batch = getBatch(i, i + batch_size, sample, None)
        embed_batch,tgt_batch = Variable(embed_batch.cuda()), Variable(tgt_batch.cuda())
        k = embed_batch.size(0)  
        # model forward
        if model_type == 'generator':
            output, _ = model((embed_batch, length))
        else:
            output = model((embed_batch, length))       
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        all_prob = output.data.cpu().numpy()
        for i in range(len(all_prob)):
            mapeach += 1/int((np.argwhere(np.argsort(-all_prob[i])==tgt_batch.data.long().cpu().numpy()[i])[0]+1))
            
    eval_acc  = np.round(100 * correct / sample.length(),2)
    eval_map=np.round(100 * mapeach / sample.length(),2)  
    if final_eval:
        print('finalgrep : accuracy {0} : {1}, map {0} : {2}'.format(eval_type, eval_acc, eval_map))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} : {2}, map {1} : {3}'.format(epoch, eval_type, eval_acc, eval_map))
    return eval_acc, eval_map 
