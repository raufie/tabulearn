# mainly for checking stuff
import gym
import torch
import controls
import matplotlib.pyplot as plt
from Demo.DemoPolicy import demo_ansi

def initializer2(nS, nA):
    q = torch.zeros((nS, nA))
def initializer(nS, nA):
    q = torch.rand((nS, nA))
    return q        
    
def main():
    n_episodes = 500
    alpha = 0.5
    eps =0.2
    gamma = 1.0
    k = 0.001

    env = gym.make("CliffWalking-v0")    
    control =controls.Sarsa(env, initializer = initializer)
    
    results = control.train(n_episodes, alpha, eps=eps, gamma = gamma)

    
    plt.plot(results[0], label="rewards per episode")
    plt.plot(results[1], label="steps per episode")
    plt.legend()
    plt.show()
    demo_ansi(control,env, limit=19)

if __name__ == "__main__":
    main()