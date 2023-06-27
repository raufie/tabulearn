from MethodBase.MethodBase import MethodBase
from tqdm import tqdm
class QLearning(MethodBase):
    """
    Q Learning is an off-policy method
    Meaning that the policy that is used to take an action (E-Greedy in this case) is not equal to the learned policy (which is greedy here)
    The policy being learned is used for Q value updates as u can see below, when we use the max Q values for the next state to get Q

    Between sarsa and this, only one difference exists... the action taken there is by the same policy when acting and when updating


    """
    def __init__(self, env, initializer):
        super().__init__(env, initializer)


    def _step(self, state:int, alpha:float, gamma:float, eps:float )->tuple:
        # # a single generalized policy iteration step
        a = self.e_greedy(state, eps)
        s_, r, done,info = self.env.step(a.item())
        self.Q[state, a] = self.Q[state, a] + alpha * (r + gamma*self.Q[s_].max() - self.Q[state, a])

        return (s_, r, done, info)

    def train(self, n_episodes:int, step_size:float, gamma = 1.0, eps = 0.0, callback = None , callback_per_episode = 500):
        _i = 0
        
        Rs = []

        steps_per_episode = []
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

            if callback is not None and (episode+1) % callback_per_episode == 0:
                callback((self.Q, Rs[_i:], steps_per_episode[_i:]))
                _i = episode
        
        return (Rs, steps_per_episode)
