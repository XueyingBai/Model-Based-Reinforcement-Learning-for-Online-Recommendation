from datetime import datetime
import os
import numpy as np
import torch

##################################################
#Get sequences from the text
##################################################
class Seq(object):
    """Instance that represent a sample of train/dev/test corpus."""

    def __init__(self, firstline,item_dic):
        words=firstline.rstrip().split("	")
        #self.usrid = int(words[2])
        self.timeSeq = [0.0]
        self.previousTime=datetime.strptime(words[4], "%Y-%m-%d %H:%M:%S")
        item_dic.add_item(int(words[3]))
        self.itemSeq = [item_dic.getItem()[int(words[3])]]
            
    def add_Seq(self, line, item_dic):
        words=line.rstrip().split("	")
        intervel=(datetime.strptime(words[4], "%Y-%m-%d %H:%M:%S")-self.previousTime).total_seconds()
        self.previousTime=datetime.strptime(words[4], "%Y-%m-%d %H:%M:%S")
        self.timeSeq.append(intervel/10e8)
        #self.timeSeq.append(intervel)
        item_dic.add_item(int(words[3]))
        self.itemSeq.append(item_dic.getItem()[int(words[3])])
        assert len(self.itemSeq) == len(self.timeSeq)
        
    def getItem(self):
        return self.itemSeq

    def getTime(self):
        return self.timeSeq

##################################################
#Split the traininig and testing parts in sequences
##################################################
class Sample(object):
    def __init__(self):
        self.timeSeq=[]
        self.itemSeq=[]
        self.purchase = []
        self.targetItem=[]
        self.targetTime=[]
      
    def genSample(self, item, time, pre_window = 0.5):
        length=len(item)
        for j in  range(length):
            seqLen=len(item[j])
            pre_len = int(pre_window * seqLen)
            if pre_len > 1:
                self.itemSeq.append(item[j][:pre_len])
                self.timeSeq.append(time[j][:pre_len])
                self.targetItem.append(item[j][pre_len:])
                self.targetTime.append(time[j][pre_len:])
                self.purchase.append(item[j][-1])
    
    def batchSample(self, batchstart, batchend):
        return np.array(self.timeSeq[batchstart:batchend]), np.array(self.itemSeq[batchstart:batchend]), np.array(self.targetItem[batchstart:batchend]), np.array(self.targetTime[batchstart:batchend]), np.array(self.purchase[batchstart:batchend])
    
    def length(self):
        return len(self.timeSeq)


class Dictionary(object):
    def __init__(self):
        self.item_dic={}
        self.current = 0
        
    def add_item(self, item_id):
        if item_id not in self.item_dic:
            self.item_dic[item_id]=self.current
            self.current += 1
            
    def getItem(self):
        return self.item_dic    
    
    def length(self):
        return len(self.item_dic)

def ReadSeq(Path, feature_path):#, PathCor):
    #corr=ReadCorres(PathCor)
    item_dic=Dictionary()
    Seqlist=[]
    for File in os.listdir(Path):
        if File.endswith(".txt"):
            fileSeq=open(os.path.join(Path,File),'r') 
            Seqline=fileSeq.readlines()
            num=0
            for line in Seqline:
                if num == 0:
                    firstSeq=Seqline[0]
                    seq=Seq(firstSeq, item_dic)
                else:
                    seq.add_Seq(line,item_dic)
                num=num+1
            Seqlist.append(seq)
    #feature_vec=FeatureMap(item_dic,corr)
    feature_vec=FeatureMap(item_dic, feature_path)
    numitem=item_dic.length()
    return Seqlist, feature_vec, numitem

#If get existed features
def FeatureMap(item_dict,feature_path):
    feature_vec={}
    feature_vec[0]=''
    item=item_dict.getItem()
    with open(feature_path) as f:
        for line in f:
            item_id, vec = line.split('	',1)
            if item_id in item:
                #item[int(item_id)] is the item index
                feature_vec[item_id] = np.array(list(map(float, vec.split())))
    return feature_vec     

def sampleSplit(train, dev, Seqlist, pre_window = 0.5):
    trainSample=Sample()
    validSample=Sample()
    testSample=Sample()
    
    itemlist=[]
    timelist=[]
    for i in Seqlist:
        itemlist.append(i.getItem())
        timelist.append(i.getTime())
    #Get length
    length=len(itemlist)
    #Get split
    trainnum=int(length*train)
    trainindex=np.random.permutation(trainnum)
    #validnum=int(length*dev)
    #validindex=np.random.permutation(np.array(range(trainnum,trainnum+validnum)))
    #testindex=np.random.permutation(np.array(range(trainnum+validnum,length)))
    testindex=np.random.permutation(np.array(range(trainnum,length)))
    #Generate Sample
    trainSample.genSample(np.array(itemlist)[trainindex].tolist(), np.array(timelist)[trainindex].tolist(), pre_window)
    #validSample.genSample(np.array(itemlist)[validindex].tolist(),np.array(timelist)[validindex].tolist(), pre_window)
    testSample.genSample(np.array(itemlist)[testindex].tolist(),np.array(timelist)[testindex].tolist(), pre_window)
    #return trainSample, validSample, testSample  
    return trainSample, testSample  

#Get one user behavior sequence
def getSeq(seq_no, samples): 
    batchTime, batchItem, batchTar_i, batchTar_t, batch_pur = samples.batchSample(seq_no, seq_no+1)
    length_pre = batchTime.shape[1]
    length_tar = batchTar_i.shape[1]
    lengths = np.array([batchTime.shape[1]])
    item=torch.LongTensor(batchItem).transpose_(0, 1).to(device)
    time=torch.FloatTensor(batchTime).transpose_(0, 1).to(device)
    TargetItem=torch.LongTensor(batchTar_i).transpose_(0, 1).to(device)
    TargetTime=torch.FloatTensor(batchTar_t).transpose_(0, 1).to(device)
    purchase = torch.LongTensor(batch_pur).to(device)
    return lengths, item, time, TargetItem, TargetTime, purchase