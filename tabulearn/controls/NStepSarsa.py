from MethodBase.MethodBase import MethodBase
from tqdm import tqdm
class NStepSarsa(MethodBase):
    """
        This is N step sarsa
        
    """
    def __init__(self, env, initializer):
        super().__init__(env, initializer)


    def _step(self, state:int, alpha:float, gamma:float, eps:float )->tuple:
        # # a single generalized policy iteration step
        pass

    def train(self, n_episodes:int, step_size:float, gamma = 1.0, eps = 0.0, callback = None , callback_per_episode = 500, n = 5):
        steps_per_episode = []
        Rs = []
        for episode in tqdm(range(n_episodes)):    
            s = self.env.reset()
            T = float('inf')
            Tau = 0
            t = 0
            
            S = [0]*(n+1)
            R = [0]*(n+1)
            A = [0]*(n+1)
            
            S[0] = s
            
            
            reward_sum = 0
            error = 0
            action = self.e_greedy(s, eps).item()
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
                        next_action = self.e_greedy(s, eps).item()
                        A[(t+1)%(n+1)] = next_action
                        
                    
                Tau = t - n + 1
                
                if Tau >= 0:
                    G = sum([gamma**(k-Tau-1)*R[(k)%(n+1)]for k in range(Tau+1, min(Tau+n, T))])
                    if Tau + n < T:
                        G = G + gamma**(n)*self.Q[S[(Tau+n)%(n+1)],A[(Tau+n)%(n+1)]]
                        
                    error =error+ (G - self.Q[S[Tau%(n+1)], A[Tau%(n+1)]])**2
                    self.Q[S[Tau%(n+1)], A[Tau%(n+1)]]+= step_size*(G - self.Q[S[Tau%(n+1)], A[Tau%(n+1)]])
                t+=1
            
        return (Rs, steps_per_episode)
