class State(object):
    
    def __init__(self):
        pass
    
    def __eq__(self, other):
        raise NotImplemented
    
    @property
    def is_game_over(self):
        raise NotImplemented
    
    def compute_reward(self):
        raise NotImplemented
        
    def get_available_actions(self):
        raise NotImplemented
    
    def get_next_state(self, action):
        raise NotImplemented
        

class Action(object):
    
    def __init__(self):
        pass