import numpy as np
import time as time
import scipy.sparse as sp
from MDP import MRP

class LP(MRP):
    def __init__(self, states, transitions, actions, w, usr_feature, discount=0.99, skip_check=False):
        try:
            from cvxopt import matrix, solvers
            self._linprog = solvers.lp
            self._cvxmat = matrix
        except ImportError:
            raise ImportError("The python module cvxopt is required to use "
                              "linear programming functionality.")
        # initialise the MRP. epsilon and max_iter are not needed
        MRP.__init__(self, states, transitions, actions, w)
        solvers.options['show_progress'] = False        
        ##Transition and reward format
        self.usr = usr_feature
        self.Trans = np.zeros((len(self.A), len(self.T), len(self.T)))
        self.Reward = np.zeros((len(self.A), len(self.T)))        
        ##Get trans and reward
        self.Get_transition()
        self.Get_reward()        
        #self.P = self.computeTransition(self.Trans)
        self.P = self.Trans
        ##Policy and value
        self.V = None
        # policy can also be stored as a vector
        self.policy = None
    
    def computeTransition(self, transition):
        print(transition[0])
        return tuple(transition[a] for a in range(self.Reward))
    
    def Get_transition(self):
        for j in range(len(self.T)):
            self.Trans[:, j, :] = self.SAS_trans(j)
                
    def Get_reward(self):
        for i in range(len(self.A)):
            for j in range(len(self.T)):
                #self.Reward[i, j] = self.Immediate_reward(self.usr, j, i)
                self.Reward[i, j] = self.Immediate_reward(i)
    
    def bellmanOperator(self, V=None):
        # Apply the Bellman operator on the value function.
        if V is None:
            V = self.V
            try:
                assert V.shape in ((len(self.T),), (1, len(self.T))), "V is not the " \
                    "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        Q = np.empty((len(self.A), len(self.T)))
        for aa in range(len(self.A)):
            Q[aa] = self.Reward[aa] + self.gamma * self.P[aa].dot(V)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        return (Q.argmax(axis=0), Q.max(axis=0))
        
    def run(self):
        # Run the linear programming algorithm.
        self.time = time.time()
        f = self._cvxmat(np.ones((len(self.T), 1)))
        h = np.array(self.Reward).reshape(len(self.T) * len(self.A), 1, order="F")
        h = self._cvxmat(h, tc='d')
        M = np.zeros((len(self.A) * len(self.T), len(self.T)))
        for aa in range(len(self.A)):
            pos = (aa + 1) * len(self.T)
            M[(pos - len(self.T)):pos, :] = (
                self.gamma * self.P[aa] - sp.eye(len(self.T), len(self.T)))
        M = self._cvxmat(M)
        self.V = np.array(self._linprog(f, M, -h)['x']).reshape(len(self.T))
        # apply the Bellman operator
        self.policy, self.V = self.bellmanOperator()
        # update the time spent solving
        self.time = time.time() - self.time
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())
        
class PolicyIteration(MRP):
    def __init__(self, states, transitions, actions, w, usr_feature, discount=0.99, policy0=None,
                 max_iter=1000, eval_type=0, skip_check=False):
        # Set up the MDP
        MRP.__init__(self, states, transitions, actions, w)
        ##Transition and reward format
        self.usr = usr_feature
        self.Trans = np.zeros((len(self.A), len(self.T), len(self.T)))
        self.Reward = np.zeros((len(self.A), len(self.T)))        
        ##Get trans and reward
        self.Get_transition()
        self.Get_reward()        
        #self.P = self.computeTransition(self.Trans)
        self.P = self.Trans
        #Specific setting for policy iteration
        self.iter = 0
        self.max_iter = max_iter
        
        ##Policy and value
        # Check if the user has supplied an initial policy. If not make one.
        if policy0 is None:
            # Initialise the policy to the one which maximises the expected
            # immediate reward
            null = np.zeros(len(self.T))
            self.policy, null = self.bellmanOperator(null)
            del null
        else:
            # Use the policy that the user supplied
            policy0 = np.array(policy0)
            # Make sure the policy is the right size and shape
            assert policy0.shape in ((len(self.T), ), (len(self.T), 1), (1, len(self.T))), \
                "'policy0' must a vector with length of #State."
            # reshape the policy to be a vector
            policy0 = policy0.reshape(len(self.T))
            # The policy can only contain integers between 0 and S-1
            msg = "'policy0' must be a vector of integers between 0 and S-1."
            assert not np.mod(policy0, 1).any(), msg
            assert (policy0 >= 0).all(), msg
            assert (policy0 < len(self.T)).all(), msg
            self.policy = policy0
        # set the initial values to zero
        self.V = np.zeros(len(self.T))
        # Do some setup depending on the evaluation type
        if eval_type in (0, "matrix"):
            self.eval_type = "matrix"
        elif eval_type in (1, "iterative"):
            self.eval_type = "iterative"
        else:
            raise ValueError("'eval_type' should be '0' for matrix evaluation "
                             "or '1' for iterative evaluation. The strings "
                             "'matrix' and 'iterative' can also be used.")
            
    def Get_transition(self):
        for j in range(len(self.T)):
            self.Trans[:, j, :] = self.SAS_trans(j)
                
    def Get_reward(self):
        for i in range(len(self.A)):
            for j in range(len(self.T)):
                #self.Reward[i, j] = self.Immediate_reward(self.usr, j, i)
                self.Reward[i, j] = self.Immediate_reward(i)

    def bellmanOperator(self, V=None):
        # Apply the Bellman operator on the value function.
        if V is None:
            V = self.V
            try:
                assert V.shape in ((len(self.T),), (1, len(self.T))), "V is not the " \
                    "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        Q = np.empty((len(self.A), len(self.T)))
        for aa in range(len(self.A)):
            Q[aa] = self.Reward[aa] + self.gamma * self.P[aa].dot(V)
        return (Q.argmax(axis=0), Q.max(axis=0)) #action, value
    
    def computePpolicyPRpolicy(self):
        # Compute the transition matrix and the reward matrix for a policy.
        # Ppolicy(SxS)  = transition matrix for policy
        # PRpolicy(S)   = reward matrix for policy
        Ppolicy = np.empty((len(self.T), len(self.T)))
        Rpolicy = np.zeros(len(self.T))
        for aa in range(len(self.A)):  
            ind = (self.policy == aa).nonzero()[0]
            # if no rows use action a, then no need to assign
            if ind.size > 0:
                try:
                    Ppolicy[ind, :] = self.P[aa][ind, :]
                except ValueError:
                    Ppolicy[ind, :] = self.P[aa][ind, :].todense() #?
                Rpolicy[ind] = self.Reward[aa][ind]
        if type(self.Reward) is sp.csr_matrix:
            Rpolicy = sp.csr_matrix(Rpolicy)
        return (Ppolicy, Rpolicy)

    def evalPolicyIterative(self, V0=0, epsilon=0.0001, max_iter=10000):
        # Evaluate a policy using iteration.
        # Vpolicy(S) = value function, associated to a specific policy
        try:
            assert V0.shape in ((len(self.T), ), (len(self.T), 1), (1, len(self.T))), \
                "'V0' must be a vector of length #State."
            policy_V = np.array(V0).reshape(len(self.T))
        except AttributeError:
            if V0 == 0:
                policy_V = np.zeros(len(self.T))
            else:
                policy_V = np.array(V0).reshape(len(self.T))

        policy_P, policy_R = self.computePpolicyPRpolicy()

        itr = 0
        done = False
        while not done:
            itr += 1

            Vprev = policy_V
            #Update V
            policy_V = policy_R + self.gamma * policy_P.dot(Vprev)

            variation = np.absolute(policy_V - Vprev).max()
            #print('epoch {0}: value_loss: {1}'.format(itr, variation))
            
            # ensure |Vn - Vpolicy| < epsilon
            if variation < ((1 - self.gamma) / self.gamma) * epsilon:
                done = True
                print("Optimal value!")
            elif itr == max_iter:
                done = True
                print("Maximum iteration!")
        self.V = policy_V
    
    def evalPolicyMatrix(self):
        # Evaluate the value function of the policy using linear equations.
        # Vpolicy(S) = value function of the policy
        Ppolicy, Rpolicy = self.computePpolicyPRpolicy()
        # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)* PR
        self.V = np.linalg.solve(
            (sp.eye(len(self.T), len(self.T)) - self.gamma * Ppolicy), Rpolicy)

    def run(self):
        # Run the policy iteration algorithm.
        self.time = time.time()

        while True:
            self.iter += 1
            # these _evalPolicy* functions will update the classes value
            # attribute
            if self.eval_type == "matrix":
                self.evalPolicyMatrix()
            elif self.eval_type == "iterative":
                self.evalPolicyIterative()
            # This should update the classes policy attribute but leave the
            # value alone
            policy_next, null = self.bellmanOperator()
            del null
            # calculate in how many places does the old policy disagree with
            # the new policy
            n_different = (policy_next != self.policy).sum()
            # if verbose then continue printing a table
            print('epoch {0}: different action: {1}'.format(self.iter, n_different))
            # Once the policy is unchanging of the maximum number of
            # of iterations has been reached then stop
            if n_different == 0:
                print("Stop because of policy unchanged")
                break
            elif self.iter == self.max_iter:
                print("Reach the maximum iteration!")
                break
            else:
                self.policy = policy_next

        self.V = tuple(self.V.tolist())
        try:
            self.policy = tuple(self.policy.tolist())
        except AttributeError:
            self.policy = tuple(self.policy)
        self.time = time.time() - self.time