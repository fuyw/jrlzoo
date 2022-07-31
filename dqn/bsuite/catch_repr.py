import numpy as np
from models import DQN

def get_all_states():
    def get_state(pos1, pos2):
        assert pos1 in range(45)
        assert pos2 in range(5)
        state = np.zeros((10, 5), dtype=np.float32)
        state[pos1//5][pos1%5] = 1.0        
        state[9][pos2] = 1.0
        return state
    states = []
    for i in range(45):
        for j in range(5):
            state = get_state(i, j)
            states.append(state)
    return states


states = get_all_states()



