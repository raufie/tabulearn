from MethodBase.MethodBase import MethodBase
from tqdm import tqdm
import numpy as np
import torch
class QSigma(MethodBase):
    """
        QSigma just alternates betwee
    """
    def __init__(self, env, initializer):
        super().__init__(env, initializer)


    def _step(self, state:int, alpha:float, gamma:float, eps:float )->tuple:
        pass

    def _pi(self, S, Q, eps):
        
        probs = np.ones(self.nA)*(eps/self.nA)
        probs[np.argmax(Q[S])] +=1-eps
        return probs

    def train(self, n_episodes:int, step_size:float, gamma = 1.0, eps = 0.0,behavior_eps=0.3, callback = None , callback_per_episode = 500, n = 5):
         
        steps_per_episode = []
        Rs = []
        for episode in tqdm(range(n_episodes)):  
            s = self.env.reset()
            T = float('inf')
            Tau = 0
            t = 0
    #         STORES
            S = [0]*(n+1)
            R = [0]*(n+1)
            SIGMA = [0]*(n+1)        
            A = [0]*(n+1)
            RHO = [0]*(n+1)
            
            S[0] = s
            
            
            reward_sum = 0
            error = 0
            action = self.e_greedy(s, behavior_eps).item()
            A[t] = action
            
            while Tau != T - 1:
                
                if t < T:
                    
                    
                    next_state, reward, done, _ = self.env.step(A[t%(n+1)])
                    
                    R[(t+1)%(n+1)] = reward
                    S[(t+1)%(n+1)] = next_state
                    s = next_state
                    
                    
                    
                    reward_sum+= reward
                    
                    
                    if done:
                        T = t+1
                        steps_per_episode.append(t)
                        Rs.append(reward_sum)
                    else:
                        next_action = self.e_greedy(s, behavior_eps).item()
                        A[(t+1)%(n+1)] = next_action
                        SIGMA[(t+1)%(n+1)] = np.random.randint(0,2)
                        RHO[(t+1)%(n+1)] = self._pi(S[(t+1)%(n+1)], self.Q, eps)/self._pi(S[(t+1)%(n+1)], self.Q, behavior_eps)
                    
                Tau = t - n + 1
                
                if Tau >= 0:
                    G = 0
                    if t+1 >=T:
                        pass
                    else:
                        G = R[(t+1)%(n+1)] + gamma* sum(torch.tensor(self._pi(S[(t+1)%(n+1)], self.Q, eps))*self.Q[S[(t+1)%(n+1)]])
                    
                    for k in range(min(t+1, T), Tau+1):
                        if k == T:
                            G = R[T%(n+1)]
                        else :
                            expected_value = sum((torch.tensor(self._pi(S[(k)%(n+1)], self.Q, eps))*self.Q[S[(k)%(n+1)]]) )
                            control = SIGMA[k%(n+1)]*RHO[k%(n+1)] + (1-RHO[k%(n+1)])*self._pi(S[k%(n+1)], self.Q, eps)[A[k%(n+1)]]
                            G = R[k%(n+1)]+ gamma*expected_value + gamma*control*(G - self.Q[S[k%(n+1)], A[k%(n+1)]])
                    self.Q[S[Tau%(n+1)], A[Tau%(n+1)]]+= step_size*(G - self.Q[S[Tau%(n+1)], A[Tau%(n+1)]])
                    error =error+ (G - self.Q[S[Tau%(n+1)], A[Tau%(n+1)]])**2
                t+=1
                
        return (Rs, steps_per_episode)
