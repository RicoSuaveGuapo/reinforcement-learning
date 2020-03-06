import gym

import numpy as np
import matplotlib.pyplot as plt

import torch.multiprocessing as mp

from net import A3C
from worker import Worker
from shared_adam import SharedAdam


def main():
    gamma = 0.9
    max_episodes = 2000
    update_global_iter = 10
    env_name = 'MountainCar-v0'

    env = gym.make(env_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    
    global_net = A3C(s_dim, a_dim)
    global_net.share_memory()
    global_opt = SharedAdam(global_net.parameters(), lr=0.001)
    
    global_episode = mp.Value('i', 0)
    global_rewards = mp.Value('d', 0.)
    result_queue = mp.Queue()

    num_cpu = mp.cpu_count()
    print('cpu count:', num_cpu)
    workers = [Worker(global_net, global_opt, global_episode, global_rewards, result_queue,
                      gamma, max_episodes, update_global_iter, env_name, i) for i in range(num_cpu)]
    [w.start() for w in workers]
    
    results = []
    while True:
        result = result_queue.get()
        if result == 'done':
            break
        else:
            results.append(result)
            
    [w.join() for w in workers]
    
    print('done')
    
    plt.plot(results)
    plt.ylabel('Moving average episode reward')
    plt.xlabel('Step')
    plt.title('A3C')
    plt.savefig('result.png')
    plt.show()

    
if __name__ == '__main__':
    main()
