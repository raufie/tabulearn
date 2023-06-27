from MethodBase.MethodBase import MethodBase
from tqdm import tqdm
import numpy as np
class ExpectedSarsa(MethodBase):
    """
        This is an ExpectedSarsa-0, an n step version also exists... u just follow normal sarsa but at the end get expected value over there      
    """
    def __init__(self, env, initializer):
        super().__init__(env, initializer)

    def _pi(self, S, Q, eps):
        
        probs = np.ones(self.nA)*(eps/self.nA)
        probs[np.argmax(Q[S])] +=1-eps
        return probs

    def _step(self, state:int, alpha:float, gamma:float, eps:float )->tuple:
        # # a single generalized policy iteration step
        a = self.e_greedy(state, eps)
        s_, r, done,info = self.env.step(a.item())

        a_ = self.e_greedy(s_, eps)

        self.Q[state, a] = self.Q[state, a] + alpha * (r + sum(self.Q[s_]*self._pi(s_, self.Q, eps)) - self.Q[state, a])

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

            # callbacks
            if callback is not None and (episode+1) % callback_per_episode == 0:
                callback((self.Q, Rs[_i:], steps_per_episode[_i:]))
                _i = episode
        return (Rs, steps_per_episode)
