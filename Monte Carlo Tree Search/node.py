import numpy as np


class Node(object):
    
    def __init__(self, state, parent=None):
        self.state = state
        
        self.parent = parent
        self.children = []
        
        self.n = 0
        self.q = 0.0
    
    @property
    def is_terminal(self):
        return self.state.is_game_over
    
    @property
    def is_fully_expanded(self):
        available_actions = self.state.get_available_actions()
        return len(self.children) == len(available_actions)
    
    def expand(self):
        assert not self.is_fully_expanded
        
        tried_states = [c.state for c in self.children]
        available_actions = self.state.get_available_actions()
        
        action = np.random.choice(available_actions)
        next_state = self.state.get_next_state(action)
        
        # it can compare two states
        # if magic method __eq__ is implemented
        while next_state in tried_states:
            available_actions.remove(action)
            action = np.random.choice(available_actions)
            next_state = self.state.get_next_state(action)
            
        child = Node(state=next_state, parent=self)
        self.children.append(child)
        
        return child
    
    def best_child(self, c_param):
        best_child = None
        best_score = -np.inf
        
        for child in self.children:
            score = child.q / child.n + c_param * np.sqrt(np.log(self.n) / child.n)
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    