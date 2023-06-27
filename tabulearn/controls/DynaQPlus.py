from MethodBase.MethodBase import MethodBase
from tqdm import tqdm
import numpy as np
import torch
class DynaQPlus(MethodBase):
    """
        DynaQ+ involves changes in the planning phase.
        Those state actions that were not taken for a long time get automatic push to their rewards...
        so longer the time elapsed, higher the automatic reward push
        every step just nudge Q[s,a] by k*sqrt(tau)
        k can be controlled as u can imagine by the expression, larger the k, larger the nudge, and more chance of taking that action in 
        the acting step.. keep it between 0-1
    """
    def __init__(self, env, initializer):
        super().__init__(env, initializer)
        self.model = {}
        self.observed_states = set({})
        self.observed_actions = {s:set() for s in range(self.env.nS)}
        self.times = torch.zeros((env.nS, env.nA), dtype=torch.int)
    def _learn(self, s,step_size,  eps= 0.2):
        a = self.e_greedy(s, eps).item()
        s_, r, done, info = self.env.step(a)
        
        self.Q[s,a] = self.Q[s,a] + step_size*(r + self.Q[s_].max() - self.Q[s, a]   )
        
        self.last_action = a
        
        return s_, r, done, info

    def _plan(self, s, a, step_size):
        r, s_ = self.model[s, a]
        self.Q[s,a] = self.Q[s,a] + step_size*(r +self.Q[s_].max() - self.Q[s, a]   )
        self.last_action = a
        return s_, r, False, None



    def _step(self, state:int, alpha:float, gamma:float, eps:float )->tuple:
        # # a single generalized policy iteration step
        a = self.e_greedy(state, eps)
        s_, r, done,info = self.env.step(a.item())

        a_ = self.e_greedy(s_, eps)

        self.Q[state, a] = self.Q[state, a] + alpha * (r + gamma*self.Q[s_, a_] - self.Q[state, a])

        return (s_, r, done, info)

    def train(self, n_episodes:int, step_size:float, gamma = 1.0, eps = 0.0, callback = None , callback_per_episode = 500, planning_steps=10, k=0.1, episode_limit=100000):
        
        self.observed_states = set({})
        self.observed_actions = {s:set() for s in range(self.env.nS)}
        curr_s = self.env.reset()
        
        rewards = []
        Rs= 0
        t = 0
        steps = []
        for t in tqdm(range(n_episodes)):
            
            done = False
            Rs = 0
            steps_this_episode = 0
            curr_s = self.env.reset()
            i = 0
            while not done:
            
                s_, r, done, _ = self._learn(curr_s, step_size, eps=eps)
                a = self.last_action
                
                
                self.model[curr_s, a] = [r, s_]


                self.observed_actions[curr_s].add(a)
                self.observed_states.add(curr_s)
                curr_s = s_
                Rs+=r
                steps_this_episode += 1
                t+=1
                self.times[curr_s][a] = t


                for n in range(planning_steps):
                    
                    s = np.random.choice(list(self.observed_states))
                    a = np.random.choice(list(self.observed_actions[s]))

                    

                    self._plan(s, a, step_size)

                    Tau = t - self.times[s][a]
                    update = k*np.sqrt(Tau)
                    update = 0.0 if np.isnan(update) else update
                    self.Q[s, a]+=(update)
                i+=1
                if i >= episode_limit:
                    done = True
            rewards.append(Rs)
            steps.append(steps_this_episode)

        return Rs, steps
