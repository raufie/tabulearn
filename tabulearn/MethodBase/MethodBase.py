import torch
import pickle
import tqdm
# -Method (holds Q tables, policies, saves policies, pi functions (for probabilies if needed), train function)

class MethodBase:
    def __init__(self, env, initializer):
        self.nS = env.nS
        self.nA = env.nA
        self.Q = initializer(env.nS, env.nA)  
        self.env = env
        self.initializer = initializer

# policies
    def e_greedy(self, state, eps, Q = None):
        if Q is None:
            Q = self.Q
        if torch.rand(1) < eps:
            return torch.randint(0,self.nA, (1,))
        else:
            return Q[state].argmax()

    
# pi's
    def _pi(self, S, Q):
        # returns probabilities of actions based on the q-values
        pass
    
    def _step(self, s, step_size, eps = None):
        # to make things cleaner it is advised to abstract away the Q value updates into this method (if possible/flexible/needed/wanted)
        pass

    def train(self, n_episodes, step_size, gamma = 1.0, eps = 0.0, callback = None):
        # gamma =  discount factor
        # return a tuple with results
        pass
    
    def _train_callback(self, data, callback = None):
        if callback is not None:
            callback(data)

# utils
    def save(self, path:str):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)
    
    def load(self, path:str):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)
