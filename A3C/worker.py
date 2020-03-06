import gym

from net import A3C
import torch.multiprocessing as mp

from utils import expand_dim, to_tensor, modify_reward, push_and_pull


class Worker(mp.Process):
    def __init__(self, global_net, global_opt, global_episode, global_rewards, result_queue,
                 gamma, max_episodes, update_global_iter, env_name, name):
        super(Worker, self).__init__()
        self.global_opt = global_opt
        self.global_net = global_net
        self.local_net = A3C(global_net.s_dim, global_net.a_dim)
        
        self.name = 'worker %d' % name
        self.global_episode = global_episode
        self.global_rewards = global_rewards
        self.result_queue = result_queue
        
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.update_global_iter = update_global_iter
        self.env = gym.make(env_name)
    
    def update_global_values(self, episode_reward):
        with self.global_episode.get_lock():
            self.global_episode.value += 1
        
        with self.global_rewards.get_lock():
            if self.global_rewards.value == 0.:
                self.global_rewards.value = episode_reward
            else:
                self.global_rewards.value = self.global_rewards.value * 0.99 + episode_reward * 0.01
        self.result_queue.put(self.global_rewards.value)
        
    def run(self):
        total_step = 1
        
        while self.global_episode.value < self.max_episodes:
            s = self.env.reset()
            
            episode_reward = 0.0
            s_list, a_list, r_list = [], [], []
            
            while True:
                s_tensor = to_tensor(expand_dim(s))
                
                a = self.local_net.choose_action(s_tensor)
                s_, r, done, _ = self.env.step(a)
                r = modify_reward(s)
                
                episode_reward += r
                
                s_list.append(s)
                a_list.append(a)
                r_list.append(r)
                
                # update the global net and its local net
                if total_step % self.update_global_iter == 0 or done: 
                    push_and_pull(self.global_net, self.local_net, self.global_opt,
                                  s_, done, s_list, a_list, r_list, self.gamma)
                    s_list, a_list, r_list = [], [], []

                    if done:
                        self.update_global_values(episode_reward)
                        print('%s: episode: %d, reward: %.2f' % (self.name,
                                                                 self.global_episode.value,
                                                                 self.global_rewards.value))
                        break
                s = s_
                total_step += 1
                
        self.result_queue.put('done')
