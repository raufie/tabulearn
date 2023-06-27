from MethodBase.MethodBase import MethodBase
from tqdm import tqdm
import numpy as np
import torch
import queue
class PrioritizedSweeping(MethodBase):
    """
       It's like DynaQ but in the planning phase we don't just sweep randomly like in basic dynaQ
       similar stuff is done in Dynamic programming... we sweep over all states randomly... mostly uniformly
       prioritized sweeping promises fast convergence by sweeping using a priority queue
       priority is roughly based on the how much of a bigger change the updates will bring
       bigger the change, bigger the priority
       only those get to be in the queue, and only those are taken from the queue to make an update
       on the intuition level, the algorithm works by working backwards from the goal states... or "backwards focusing" in planning
       instead of just doing random one... so imagine, the highest priority item gets to update the q values, then we take the neighbors of that s
       and see which ones are promising (p>theta), and them to the queue and repeat... pretty cool eh
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




    def train(self, n_episodes:int, step_size:float, gamma = 1.0, eps = 0.0, callback = None , callback_per_episode = 500, episode_limit=100000, thres=0.25):
        
        
        HASH_TO_STATE= {}

        PQueue = queue.PriorityQueue()
        curr_s = self.env.reset()
        
      
        
        rewards = []
        steps_per_episode = []

        
        curr_s = self.env.reset()
        
        for t in tqdm(range(n_episodes)):
            Rs= 0
            steps = 0
            done = False
            while not done:       
                s_, r, done, _ = self._learn(curr_s, step_size, eps=eps)
                
                a = self.last_action
                self.times[curr_s][a] = t
                
                self.model[curr_s, a] = [r, s_]
                HASH_TO_STATE[hash((curr_s, a))] = (curr_s, a)
                
                P = abs(r + gamma*self.Q[s_].max() - self.Q[curr_s][a])
                
                if P > thres:
                    PQueue.put( (P, hash((curr_s, a))))
                            
                self.observed_actions[curr_s].add(a)
                self.observed_states.add(curr_s)
                curr_s = s_
                Rs+=r
                steps+=1


              
                
                while not PQueue.empty():
                    p, hash_ = PQueue.get()
                    s,a = HASH_TO_STATE[hash_]
                    self._plan(s,a, step_size)
        #             all the values that lead to s
                    PREDECESSORS = []
                    for key in self.model:
                        if self.model[key][1] == s:
                            PREDECESSORS.append(key)
                                
                    for s__, a__ in PREDECESSORS:
                        R_, _ = self.model[s__, a__]
                        P = abs(R_ + gamma*self.Q[s].max() - self.Q[s__][a__])
                        if P > thres:
                            PQueue.put( (P, hash((s__, a__))))
                            # !beware: here we need to make sure that the highest theta gets to replace all other (same values)
                            # i haven't done it here, becuz unless i abstract it out, it seems computationally expensive
                            # idk if any python method exists... a custom PQueue Implementation is needed (maybe in later version's ill see)
                
                if steps >= episode_limit:
                    done = True    
                   

            rewards.append(Rs)
            steps_per_episode.append(steps)
            
                
                
             
        return rewards, steps_per_episode