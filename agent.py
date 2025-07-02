import random
import json
from collections import defaultdict
from environment import Direction 
import config 

class SharedQPolicy:
    def __init__(self):
        # Load hyperparameters from the config file
        self.alpha = config.ALPHA
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON
        self.epsilon_decay = config.EPSILON_DECAY
        self.min_epsilon = config.MIN_EPSILON
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.new_state_actions_explored = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(list(Direction))
        if not self.q_table[state]:
            return random.choice(list(Direction))
        max_q = max(self.q_table[state].values())
        return random.choice([a for a, q in self.q_table[state].items() if q == max_q])

    def update_q_table(self, state, action, reward, next_state):
        if action not in self.q_table[state]:
            self.new_state_actions_explored += 1
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_next_q - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filename=config.Q_TABLE_FILE_NAME):
        serializable = {str(k): {act.name: v for act, v in acts.items()} for k, acts in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=4)
        print(f"Saved shared Q-table to {filename}")

    def load_q_table(self, filename=config.Q_TABLE_FILE_NAME):
        with open(filename, 'r') as f:
            serializable = json.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float))
        for k, acts in serializable.items():
            state_tuple = eval(k)
            self.q_table[state_tuple] = defaultdict(float, {Direction[name]: v for name, v in acts.items()})
        print(f"Loaded shared Q-table from {filename}")