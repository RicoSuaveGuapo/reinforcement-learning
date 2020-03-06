import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from common import State, Action


IMAGE_DIR = '/Users/william/Documents/Programming/AI/Practice/RL/Monte Carlo Tree Search/games/images'


class TicTacToeState(State):
    
    def __init__(self, state, first_move):
        if state.ndim != 2 or state.shape[0] != state.shape[1]:
            raise ValueError("Please play on 2D square board")
        if first_move not in ['o', 'x']:
            raise ValueError("The argument `first_move` should be either 'o' or 'x'")
            
        self.board = state
        self.board_size = state.shape[0]
        self.first_move = first_move
        
    def __eq__(self, other):
        return np.allclose(self.board, other.board)
    
    @property
    def game_result(self):
        rowsum = np.sum(self.board, axis=0)
        colsum = np.sum(self.board, axis=1)
        diagsum = self.board.trace()
        antidiagsum = self.board[::-1].trace()
        
        if any(rowsum == self.board_size) or any(colsum == self.board_size) \
           or diagsum == self.board_size or antidiagsum == self.board_size:
            result = 'o wins the game'
        elif any(rowsum == -self.board_size) or any(colsum == -self.board_size) \
           or diagsum == -self.board_size or antidiagsum == -self.board_size:
            result = 'x wins the game'
        elif np.all(self.board != 0):
            result = 'tie'
        else:
            result = None
            
        return result
    
    @property
    def is_game_over(self):
        return self.game_result != None
    
    def compute_reward(self):
        result = self.game_result
        
        assert result != None
        
        if 'o' in result:
            return 1
        elif 'x' in result:
            return -1
        elif result == 'tie':
            return 0
    
    def get_available_actions(self):
        if np.sum(self.board) == 1:
            turn = 'x'
        elif np.sum(self.board) == 0:
            turn = self.first_move
        else:
            turn = 'o'
        
        indices = np.where(self.board == 0)
        actions = [TicTacToeAction(x, y, turn) for x, y in zip(indices[1], indices[0])]
        
        return actions
        
    def get_next_state(self, action):
        x, y, turn = action.x, action.y, action.turn
        
        board = np.copy(self.board)
        board[y, x] = turn
        
        return TicTacToeState(board, self.first_move)
    
    def show(self, return_array=False):
        board = np.array(Image.open(os.path.join(IMAGE_DIR, 'board.png')))
        o = np.array(Image.open(os.path.join(IMAGE_DIR, 'o.png')))
        x = np.array(Image.open(os.path.join(IMAGE_DIR, 'x.png')))
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == 0:
                    pass
                else:
                    mark = o if self.board[row, col] == 1 else x
                    size = mark.shape[0]
                    x_min = 25 + col * 80
                    x_max = x_min + size
                    y_min = 25 + row * 80
                    y_max = y_min + size
                    board[y_min:y_max, x_min:x_max] = mark
        
        if return_array:
            return board
        else:
            plt.imshow(board)
            plt.axis('off')


class TicTacToeAction(Action):
    
    def __init__(self, x, y, turn):
        self.x = x
        self.y = y
        self.turn = 1 if turn == 'o' else -1
        
    def __repr__(self):
        return 'x: {} y: {} turn: {}'.format(self.x, self.y, self.turn)


if __name__ == '__main__':
    init_state = np.zeros((3, 3))
    game = TicTacToeState(init_state, first_move='o')
    
    i = 0
    rols, cols = 1, 10
    plt.figure(figsize=(8, 8))

    while True:
        i += 1
        board = game.show(return_array=True)
        plt.subplot(rols, cols, i)
        plt.imshow(board)
        plt.axis('off')
        
        if game.is_game_over:
            break

        available_actions = game.get_available_actions()
        action = np.random.choice(available_actions)
        game = game.get_next_state(action)
        
    plt.show()
