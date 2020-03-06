import numpy as np

import torch
from torch import nn


def expand_dim(array):
    array = array.astype(np.float32)
    return np.expand_dims(array, axis=0)
    
    
def to_tensor(array):
    return torch.from_numpy(array)


def modify_reward(state):
    x, v = state
    
    reward = 50 * v**2
    
    if x > 0.20:
        reward += 1
    if x > 0.30:
        reward += 1
    if x > 0.40:
        reward += 1
    if x > 0.45:
        reward += 1
            
    return reward


def push_and_pull(global_net, local_net, opt, s_, done, s_list, a_list, r_list, gamma):
    if done:
        v_s_ = 0.0
    else:
        s_ = expand_dim(s_)
        s_ = to_tensor(s_)
        _, v_s_ = local_net.forward(s_)
        v_s_ = v_s_.data.numpy()[0, 0]

    v_target_list = []
    for r in r_list[::-1]:
        v_s_ = r + gamma * v_s_
        v_target_list.append(v_s_)
    v_target_list.reverse()
    
    s_tensor = to_tensor(np.vstack(s_list).astype(np.float32))
    a_tensor = to_tensor(np.array(a_list).astype(np.float32))
    v_target = np.array(v_target_list).astype(np.float32)
    v_target = np.expand_dims(v_target, axis=1)
    v_tensor = to_tensor(v_target)
    
    loss = local_net.loss(s_tensor, a_tensor, v_tensor)

    opt.zero_grad()
    loss.backward()
    
    for global_parameter, local_parameter in zip(global_net.parameters(), local_net.parameters()):
        global_parameter._grad = local_parameter.grad
    
    opt.step()

    local_net.load_state_dict(global_net.state_dict())
