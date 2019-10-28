#Time not included, no dictionary
import os
import numpy as np

class Seq(object):
    def __init__(self):
        self.click = []
        self.reward = []
        self.action = []
        self.label = 1 # By default it's a real sequence
            
    def add_click(self, line_click, itemlist):
        words=line_click.rstrip().split(" ")
        for i in range(len(words)):
            self.click.append(int(words[i])) 
             
            if words[i] not in itemlist:
                itemlist[words[i]]=1
            else:
                itemlist[words[i]]+=1     
             
        return itemlist 
                  
    def add_reward(self, line_reward):
        words=line_reward.rstrip().split(" ")
        for i in range(len(words)):
            self.reward.append(float(words[i]))
            
    def add_action(self, line_action, itemlist):
        words=line_action.rstrip().split(" ")
        for i in range(len(words)):
            recom = words[i].split(",")
            self.action.append(list(map(int, recom)))
            for j in recom:
                if j not in itemlist:
                    itemlist[j]=0
                '''
                else:
                    itemlist[j] += 1
                '''
        return itemlist
            
    def add_label(self, label):
        self.label = int(label)
            
class Sample(object):
    def __init__(self):
        self.itemSeq=[]
        self.target=[]
        self.action=[]
        self.reward = [] #target for rewards, like purchase
        self.clicked = {}
    
    #Generate sample for the next-click prediction training  
    def genSample_pred(self, item, reward, action, warmup, real_num_label, rec_len, add_end=True):
        length=len(item)
        for j in range(length):
            seqLen=len(item[j])
            #Filter short sequences --preprocessing
            #pre_len = int(warmup * seqLen)
            pre_len = 1
            tar_len = seqLen - pre_len
            if pre_len > 0 and tar_len > 0:
                for i in range(pre_len, seqLen):
                    self.itemSeq.append(item[j][:i])
                    self.target.append(item[j][i])
                    self.reward.append(reward[j][i]) #reward for the next item
                    if add_end:
                        self.action.append(action[j][i]+[real_num_label-1])
                    else:
                        self.action.append(action[j][i])
                    if item[j][i-1] not in self.clicked:
                        self.clicked[item[j][i-1]] = 0
                if add_end:
                    self.itemSeq.append(item[j][:seqLen])
                    self.target.append(real_num_label-1)
                    temp = np.random.randint(real_num_label-1,size = rec_len).tolist()
                    #temp = [0]
                    #temp = ((real_num_label-1)*np.ones(rec_len + 1)).tolist()
                    temp.append(real_num_label-1)
                    self.action.append(temp)
                    self.reward.append(0) #If eos, reward=0
        if add_end:
            self.clicked[real_num_label-1] = 0
                                            
    #Generate test sequences to test the next-click prediction accuracy       
    #How to deal with the recommendation list in the test?    
    def genSample_predtest(self, item, reward, action, window, real_num_label, add_end=False):
        length=len(item)
        for j in range(length):
            seqLen=len(item[j])
            #winLen=int(window*seqLen)
            winLen=1
            if winLen > 0:
                for k in range(winLen, seqLen):
                    if len(action[j][k]) >= 10:
                    #if np.unique(np.array(action[j][k])) >= 10:
                        self.itemSeq.append(item[j][:k])
                        self.target.append(item[j][k])
                        if add_end:
                            self.action.append(action[j][k]+[real_num_label-1])
                        else:
                            self.action.append(action[j][k])
                        self.reward.append(reward[j][k]) 
                ''' 
                if add_end:
                    self.itemSeq.append(item[j][:seqLen])
                    self.target.append(real_num_label-1)
                    temp = np.random.randint(real_num_label-1,size = 10).tolist()
                    temp.append(real_num_label-1)
                    self.action.append(temp) 
                    self.reward.append(0) #If eos, reward=0
                '''               
    #Pre_window can be altered for a new roll-out method(not applicable here)
    #Generate given sequences for generator to generate the rest    
    def genSample_generator(self, item, reward, action, real_num_label, rec_len, add_end=False):
        length=len(item)
        for j in range(length):
            #add_end is true for training of both user behavior model and the agent model
            if add_end:
                self.itemSeq.append(item[j]+[real_num_label-1])
                self.reward.append(reward[j]+[0])
                tmp_action = action[j]
                tmp_action.append(np.random.randint(real_num_label-1,size = rec_len).tolist())
                #tmp_action.append([real_num_label-1])
                self.action.append(tmp_action)
                #print(len(tmp_action))
            else:
                self.itemSeq.append(item[j])
                self.reward.append(reward[j])
                self.action.append(action[j])
                #print(len(action[j]))
            self.target.append([-1])    
                               
    def genSample_discriminator(self, item, reward, action, label, real_num_label, add_end=False):
        length=len(item)
        rec_len = len(action[0][0])
        for j in range(length):
            if add_end:
                self.itemSeq.append(item[j]+[real_num_label-1])
                self.reward.append(reward[j]+[0])
                tmp_action = action[j]
                tmp_action.append(np.random.randint(real_num_label-1,size = rec_len).tolist())
                self.action.append(tmp_action)
            else:
                self.itemSeq.append(item[j]) 
                self.action.append(action[j])
                #print(len(action[j][0]))
                self.reward.append(reward[j])   
            self.target.append(label[j]) 
    
    def batchSample(self, batchstart, batchend):
        return np.array(self.itemSeq[batchstart:batchend]), np.array(self.target[batchstart:batchend]), np.array(self.reward[batchstart:batchend]), np.array(self.action[batchstart:batchend])
        
    def subSample_copy(self, subnum, origin, shuffle = None):
        if shuffle == None:
            index=np.random.permutation(origin.length()) 
            self.itemSeq = np.array(origin.itemSeq)[index[:subnum]].tolist()
            self.reward = np.array(origin.reward)[index[:subnum]].tolist()
            self.action = np.array(origin.action)[index[:subnum]].tolist()
            self.target = np.array(origin.target)[index[:subnum]].tolist()
        else:
            self.itemSeq = np.array(origin.itemSeq)[shuffle[:subnum]].tolist()
            self.reward = np.array(origin.reward)[shuffle[:subnum]].tolist()
            self.action = np.array(origin.action)[shuffle[:subnum]].tolist()
            self.target = np.array(origin.target)[shuffle[:subnum]].tolist()
        
    def length(self):
        return len(self.itemSeq)
                        
def ReadSeq(seqpath, rewardpath, actionpath, labelpath = None):
    itemlist={}
    #itemlist['0'] = 0 #PAD
    Seqlist=[]
    ClickSeq=open(os.path.join('',seqpath),'r') 
    RewardSeq=open(os.path.join('',rewardpath),'r') 
    ActionSeq=open(os.path.join('',actionpath),'r')
    Clickline=ClickSeq.readlines()
    Rewardline=RewardSeq.readlines()
    Actionline=ActionSeq.readlines()
    #Labels for discriminator
    if labelpath is not None:
        LabelSeq=open(os.path.join('',labelpath),'r')
        Labelline=LabelSeq.readlines() 
    lines = len(Clickline)
    for i in range(lines):
        seq=Seq()
        itemlist = seq.add_click(Clickline[i], itemlist)
        seq.add_reward(Rewardline[i])
        itemlist = seq.add_action(Actionline[i], itemlist)
        if labelpath is not None:
            seq.add_label(Labelline[i])
        assert len(seq.action) == len(seq.click)
        Seqlist.append(seq)
    return Seqlist, itemlist
                       
def sampleSplit(trainindex, validindex, testindex, Seqlist, real_num_label, rec_len, sample_type='pretrain_gen', warm_up = 0.5):
    trainSample=Sample()
    validSample=Sample()
    testSample=Sample()
    
    clicklist=[]
    rewardlist=[]
    actionlist=[]
    targetlist=[]
    for i in Seqlist:
        clicklist.append(i.click)
        rewardlist.append(i.reward)
        actionlist.append(i.action)
        if sample_type == 'dis':
            targetlist.append(i.label)  
                      
    #Generate Sample
    if sample_type=='pretrain_gen':
        trainSample.genSample_pred(np.array(clicklist)[trainindex].tolist(), np.array(rewardlist)[trainindex].tolist(), np.array(actionlist)[trainindex].tolist(), warm_up, real_num_label, rec_len, add_end = True)
        validSample.genSample_predtest(np.array(clicklist)[validindex].tolist(),np.array(rewardlist)[validindex].tolist(), np.array(actionlist)[validindex].tolist(), warm_up, real_num_label, add_end = False)
        testSample.genSample_predtest(np.array(clicklist)[testindex].tolist(),np.array(rewardlist)[testindex].tolist(), np.array(actionlist)[testindex].tolist(), warm_up, real_num_label, add_end = False)
        
    elif sample_type=='adv':
        trainSample.genSample_generator(np.array(clicklist)[trainindex].tolist(), np.array(rewardlist)[trainindex].tolist(), np.array(actionlist)[trainindex].tolist(), real_num_label, rec_len, add_end = True)
        validSample.genSample_predtest(np.array(clicklist)[validindex].tolist(),np.array(rewardlist)[validindex].tolist(), np.array(actionlist)[validindex].tolist(), warm_up, real_num_label, add_end = False)
        testSample.genSample_predtest(np.array(clicklist)[testindex].tolist(),np.array(rewardlist)[testindex].tolist(), np.array(actionlist)[testindex].tolist(), warm_up, real_num_label, add_end = False)
        
    elif sample_type=='gen': #Not adding eos
        trainSample.genSample_generator(np.array(clicklist)[trainindex].tolist(), np.array(rewardlist)[trainindex].tolist(), np.array(actionlist)[trainindex].tolist(), real_num_label, False)
        validSample.genSample_generator(np.array(clicklist)[validindex].tolist(), np.array(rewardlist)[validindex].tolist(), np.array(actionlist)[validindex].tolist(), real_num_label, False)
        testSample.genSample_generator(np.array(clicklist)[testindex].tolist(), np.array(rewardlist)[testindex].tolist(), np.array(actionlist)[testindex].tolist(), real_num_label, False)
                
    elif sample_type=='dis':
        trainSample.genSample_discriminator(np.array(clicklist)[trainindex].tolist(), np.array(rewardlist)[trainindex].tolist(), np.array(actionlist)[trainindex].tolist(), np.array(targetlist)[trainindex].tolist(), real_num_label)
        validSample.genSample_discriminator(np.array(clicklist)[validindex].tolist(), np.array(rewardlist)[validindex].tolist(), np.array(actionlist)[validindex].tolist(), np.array(targetlist)[validindex].tolist(), real_num_label)
        testSample.genSample_discriminator(np.array(clicklist)[testindex].tolist(), np.array(rewardlist)[testindex].tolist(), np.array(actionlist)[testindex].tolist(), np.array(targetlist)[testindex].tolist(), real_num_label)
    #return trainSample, testSample
    return trainSample, validSample, testSample  

    