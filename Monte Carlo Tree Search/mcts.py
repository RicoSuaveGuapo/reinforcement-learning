import numpy as np
from node import Node


class MonteCarloTreeSearch:
    
    def __init__(self, c_param, computational_budge, value_function=None, rollout_policy=None):
        self.c_param = c_param
        self.value_function = value_function
        self.rollout_policy = rollout_policy
        self.computational_budge = computational_budge
    
    def search(self, state):
        root = Node(state)
        
        histories = []
        for _ in range(self.computational_budge):
            v = self.tree_policy(root)
            reward, rollout_history = self.rollout(v)
            node_history = self.backup(v, reward)
            
            history = node_history + rollout_history
            histories.append(history)
        
        return root.best_child(c_param=0.0), histories
    
    def tree_policy(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = node.best_child(c_param=self.c_param)
            else:
                return node.expand()
            
        return node
            
    def rollout(self, node):
        state = node.state
        history = []
        
        # if value_function is given, estimate the state directly,
        # else perform rollout
        if self.value_function is not None:
            return self.value_function(state)
        else:
            while not state.is_game_over:
                available_actions = state.get_available_actions()

                if self.rollout_policy is None:
                    action = np.random.choice(available_actions)
                else:
                    # if rollout_policy is given
                    # choose action by rollout_policy
                    action = self.rollout_policy(current_node.state)

                state = state.get_next_state(action)
                history.append((state, 'rollout'))

            return state.compute_reward(), history
    
    def backup(self, node, reward):
        history = []
        while node is not None:       
            node.n += 1
            node.q += reward
            
            history.append((node, 'n: %d, q: %.1f' % (node.n, node.q)))
            
            node = node.parent
        
        history = history[::-1]
        
        return history