import time
import os
def demo_ansi(control,env, limit=30,refresh_rate = 0.5):
    env.reset()
    out = env.render(mode="ansi")
    done = False
    s = env.reset()
    i = 0
    while not done:
        
        s_, r, done, prop = env.step(int(control.e_greedy(s, 0.0)))
        time.sleep(refresh_rate)
        os.system("cls")
        print(env.render())
        s = s_
        i+=1
        if i >= limit:
            done = True