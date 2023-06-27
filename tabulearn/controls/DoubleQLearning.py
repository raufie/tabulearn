from MethodBase.MethodBase import MethodBase
from tqdm import tqdm
import numpy as np
class DoubleQLearning(MethodBase):
    """
        This is an ExpectedSarsa-0, an n step version also exists... u just follow normal sarsa but at the end get expected value over there      
    """
    def __init__(self, env, initializer):
        self.nS = env.nS
        self.nA = env.nA
        self.Q1 = initializer(env.nS, env.nA) 
        self.Q2 = initializer(env.nS, env.nA)
        self.env = env


    def e_greedy(self, state, eps, Q = None):
        if Q is None:
            self.Q = self.Q1+self.Q2
            return super().e_greedy(state, eps, self.Q)
        else:
            return super().e_greedy(state, eps, Q)
    def _pi(self, S, Q, eps):
        
        probs = np.ones(self.nA)*(eps/self.nA)
        probs[np.argmax(Q[S])] +=1-eps
        return probs

    def _step(self, state:int, alpha:float, gamma:float, eps:float )->tuple:
        # # a single generalized policy iteration step
        a = self.e_greedy(state, eps, Q = self.Q1 + self.Q2)

        s_, r, done,info = self.env.step(a.item())

        a_ = self.e_greedy(s_, eps, Q = self.Q1 + self.Q2)

        if np.random.rand() < 0.5:
            self.Q1[state, a] = self.Q1[state, a] + alpha * (r + gamma*self.Q2[s_,self.Q1[s_].argmax()] - self.Q1[state, a])
        else:
            self.Q2[state, a] = self.Q2[state, a] + alpha * (r + gamma*self.Q1[s_,self.Q2[s_].argmax()] - self.Q2[state, a])

        return (s_, r, done, info)

    
    def train(self, n_episodes:int, step_size:float, gamma = 1.0, eps = 0.0, callback = None , callback_per_episode = 500):
        Rs = []
        steps_per_episode = []
        _i = 0
        for episode in tqdm(range(n_episodes),desc="Learning a policy"):
            
            
            s = self.env.reset()
            done = False
            r_i = 0
            n_steps = 0
            while not done:
               
                s_, r, done, info = self._step(s,step_size, gamma, eps = eps)
                s = s_
                r_i+=r
                n_steps += 1
            Rs.append(r_i)
            steps_per_episode.append(n_steps)
            
            self.Q = self.Q1 + self.Q2

            if callback is not None and (episode+1) % callback_per_episode == 0:
                callback((self.Q, Rs[_i:], steps_per_episode[_i:]))
                _i = episode
        return (Rs, steps_per_episode)
