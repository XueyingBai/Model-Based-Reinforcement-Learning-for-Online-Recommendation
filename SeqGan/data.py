#Time not included, no dictionary
import os
import numpy as np

class Seq_gen(object):
    def __init__(self):
        self.click = []
        self.purchase = []
            
    def add_click(self, line_click, itemlist):
        words=line_click.rstrip().split(" ")
        for item in words:
            self.click.append(int(item))
            if item not in itemlist:
                itemlist[item]=0
            
    def add_purchase(self, line_pur, itemlist):
        words=line_pur.rstrip().split(" ")
        for item in words:
            self.purchase.append(int(item))
            if item not in itemlist:
                itemlist[item]=0
        
    def getClick(self):
        return self.click

    def getPurchase(self):
        return self.purchase
    
class Seq_dis(object):
    def __init__(self):
        self.click = []
        self.target = 0
            
    def add_click(self, line_click, itemlist):
        words=line_click.rstrip().split(" ")
        for item in words:
            self.click.append(int(item))
            if item not in itemlist:
                itemlist[item]=0
            
    def add_target(self, line_tar, itemlist):
        words=line_tar.rstrip().split(" ")
        self.target=int(words[0])
        
    def getClick(self):
        return self.click

    def getTarget(self):
        return self.target
    
class Sample(object):
    def __init__(self):
        self.itemSeq=[]
        self.purchase = []
        self.targetItem=[]
      
    def genSample_pretrain(self, item, purchase, warmup):
        length=len(item)
        for j in  range(length):
            seqLen=len(item[j])
            pre_len = int(warmup * seqLen)
            tar_len = seqLen - pre_len
            if pre_len > 0 and tar_len > 0:
                for i in range(pre_len, seqLen):
                    self.itemSeq.append(item[j][:i])
                    self.targetItem.append(item[j][i])
                    self.purchase.append(purchase[j]) #purchase will not be used during the pre-training
                
    #pre_window can be altered for a new roll-out method
    def genSample_adversarial(self, item, purchase, pre_window):
        length=len(item)
        for j in  range(length):
            seqLen=len(item[j])
            pre_len = int(pre_window * seqLen)
            tar_len = seqLen - pre_len
            if pre_len > 0 and tar_len > 0:
                self.itemSeq.append(item[j][:pre_len])
                #self.targetItem.append(item[j][pre_len:]) #target item will not be used during the adversarial training
                self.targetItem.append(item[j][pre_len])#target item will not be used during the adversarial training
                self.purchase.append(purchase[j])
                
    def genSample_test(self, item, purchase, window):
        length=len(item)
        for j in  range(length):
            seqLen=len(item[j])
            winLen=int(window*seqLen)                  
            if winLen > 0:
                for k in range(winLen, seqLen):
                    self.itemSeq.append(item[j][:k])
                    self.targetItem.append(item[j][k])
                    self.purchase.append(purchase[j])
                        
    def genSample_dis(self, item, target):
        length=len(item)
        for j in range(length):
            self.itemSeq.append(item[j])
            self.targetItem.append(target[j])  
            self.purchase.append([0]) #Should be -1 after synchornization          
    
    def batchSample(self, batchstart, batchend):
        return np.array(self.itemSeq[batchstart:batchend]), np.array(self.targetItem[batchstart:batchend]), np.array(self.purchase[batchstart:batchend])
        
    def subSample_copy(self, subnum, origin):
        index=np.random.permutation(origin.length()) 
        self.itemSeq = np.array(origin.itemSeq)[index[:subnum]].tolist()
        self.purchase = np.array(origin.purchase)[index[:subnum]].tolist()
        self.targetItem = np.array(origin.targetItem)[index[:subnum]].tolist()
        
    def length(self):
        return len(self.itemSeq)
                        
def ReadSeq_Gen(clickpath, purchasepath):
    itemlist={}
    Seqlist=[]
    ClickSeq=open(os.path.join('',clickpath),'r') 
    PurchaseSeq=open(os.path.join('',purchasepath),'r') 
    Clickline=ClickSeq.readlines()
    Purchaseline=PurchaseSeq.readlines()
    lines = len(Clickline)
    for i in range(lines):
        seq=Seq_gen()
        seq.add_click(Clickline[i], itemlist)
        seq.add_purchase(Purchaseline[i], itemlist)
        Seqlist.append(seq)
    numitem=len(itemlist)
    print(numitem)
    return Seqlist, numitem

def ReadSeq_Dis(seqpath, targetpath):
    itemlist={}
    Seqlist=[]
    ClickSeq=open(os.path.join('',seqpath),'r') 
    Target=open(os.path.join('',targetpath),'r') 
    Clickline=ClickSeq.readlines()
    targetline=Target.readlines()
    lines = len(Clickline)
    for i in range(lines):
        seq=Seq_dis()
        seq.add_click(Clickline[i], itemlist)
        seq.add_target(targetline[i], itemlist)
        Seqlist.append(seq)
    return Seqlist
                       
def sampleSplit(trainindex, validindex, testindex, Seqlist, sample_type='pretrain_gen', warm_up = 0.5, pre_window = 0.5):
    trainSample=Sample()
    validSample=Sample()
    testSample=Sample()
    
    clicklist=[]
    purchaselist=[]
    targetlist=[]
    for i in Seqlist:
        clicklist.append(i.getClick())
        if sample_type == 'dis':
            targetlist.append(i.getTarget())
        else:
            purchaselist.append(i.getPurchase())
    #Generate Sample
    if sample_type=='pretrain_gen':
        trainSample.genSample_pretrain(np.array(clicklist)[trainindex].tolist(), np.array(purchaselist)[trainindex].tolist(), warm_up)
        validSample.genSample_pretrain(np.array(clicklist)[validindex].tolist(),np.array(purchaselist)[validindex].tolist(), warm_up)
        testSample.genSample_test(np.array(clicklist)[testindex].tolist(),np.array(purchaselist)[testindex].tolist(), warm_up)
        
    elif sample_type=='adv':
        trainSample.genSample_adversarial(np.array(clicklist)[trainindex].tolist(), np.array(purchaselist)[trainindex].tolist(), pre_window)
        validSample.genSample_pretrain(np.array(clicklist)[validindex].tolist(),np.array(purchaselist)[validindex].tolist(), pre_window)
        testSample.genSample_test(np.array(clicklist)[testindex].tolist(),np.array(purchaselist)[testindex].tolist(), pre_window)
        
    elif sample_type=='dis':
        trainSample.genSample_dis(np.array(clicklist)[trainindex].tolist(), np.array(targetlist)[trainindex].tolist())
        validSample.genSample_dis(np.array(clicklist)[validindex].tolist(),np.array(targetlist)[validindex].tolist())
        testSample.genSample_dis(np.array(clicklist)[testindex].tolist(),np.array(targetlist)[testindex].tolist())
    #return trainSample, testSample
    return trainSample, validSample, testSample  

    