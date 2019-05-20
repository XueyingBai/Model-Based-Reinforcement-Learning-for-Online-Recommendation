import numpy as np
import random

def GenerateMatrix(num, fea_len):
    return np.random.rand(num, fea_len)

def GenerateGraph(num, left=0.4, right=0.7):
    #the transition matrix
    graph = np.random.randn(num, num)
    for i in range(num):
        sumr=0
        for j in range(num):
            '''
            if left<=graph[i,j] and graph[i,j]<=right:
                graph[i,j]=0
            '''
            sumr+=graph[i,j]
        #normalize
        graph[i,:]=graph[i,:]/sumr
    end=random.sample(range(num),1)
    graph[end,:]=0*graph[end,:]
    return graph, end
    
class MRP(object):    
    """ states: representation of each state
        transition: state-state transition probabilities
        actions: representation of each action """ 
    
    def __init__(self, states, transitions, actions, w, discount=0.99):   
        self.S = np.matrix(states)
        self.T = np.matrix(transitions)
        self.A = np.matrix(actions)
        self.w = np.matrix(w)
        self.gamma = discount
    ''' 
    def Immediate_reward(self, usr_fea, s_i, a_k):
        concat = np.append(self.A[a_k,:], np.matrix(usr_fea), axis=1)
        assert self.S[s_i].shape[1] == concat.shape[1]
        return self.S[s_i] * concat.transpose()
    ''' 
    def Immediate_reward(self, a_k):
        return a_k
         
    def SAS_trans(self, s_i):
        act_len=len(self.A)
        state_len=len(self.T)
        trans=np.zeros([act_len, state_len])
        for i in range(act_len):
            summ=0
            for j in range(state_len):
                if self.T[s_i, j] > 0:
                    trans[i,j]=np.exp(self.T[s_i,j]*(self.w*self.A[i].transpose()))
                    summ += trans[i,j]
                else:
                    trans[i,j] = 0
            if summ > 0:
                trans[i,:]=trans[i,:]/summ
        return trans    
    
    def Get_state_num(self):
        return self.S.shape[0]
    
    def Get_state(self, s_i):
        return self.S[s_i]
        
def Environment(num_states, fea_states, fea_actions, num_actions, usr_num, usr_fea):
    states=GenerateMatrix(num_states, fea_states)
    actions=GenerateMatrix(num_actions, fea_actions)
    transitions, end=GenerateGraph(num_states, 0.3, 0.7)
    w=GenerateMatrix(1, fea_actions)
    usr_fea_vec=GenerateMatrix(usr_num, usr_fea)
    Env=MRP(states, transitions, actions, w)
    #print(transitions)
    return Env, usr_fea_vec, end