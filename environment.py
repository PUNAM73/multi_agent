import random
from collections import defaultdict
from enum import Enum
import config  # Import our configuration file

# --- Enums for Readability ---
class LogKeys(Enum):
    EPISODES = 1
    TOTAL_STEPS = 2
    EPISODE_COLLISIONS = 3
    EPISODE_DELIVERIES = 4
    EPSILON = 5
    HEAD_ON_COLLISION = 6
    DELIVERIES = 7
    AGENT_STEPS = 8
    WALL_COLLISION = 9
    NEW_STATE_ACTION = 10

class Direction(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)

class Reward(Enum):
    # Recommended balanced rewards
    SUCCESSFUL_DELIVERY = +5000.0
    HEAD_ON_COLLISION = -10000.0 # Only for collision-possible env
    YIELD_PENALTY = -500.0
    WALL_COLLISION = -50.0  # Make this worse than yielding
    PROGRESS_REWARD = +50.0
    PICKUP_INCENTIVE = +100.0
    STEP_PENALTY = -100.0

# --- Environment Class ---
class GridWorld:
    def __init__(self, size=config.GRID_SIZE, num_agents=config.NUM_AGENTS):
        self.size = size
        self.num_agents = num_agents
        self.pickup_loc = (0, 0)
        self.dropoff_loc = (size - 1, size - 1)
        self.training_phase = 1
        self._pos_to_agent_id = {}
        self.reset()

    def set_training_phase(self, phase):
        print(f"[GridWorld] Setting training phase to {phase}")
        self.training_phase = phase

    def reset(self):
        possible_positions = [(i, j) for i in range(self.size) for j in range(self.size) if
                              (i, j) not in [self.pickup_loc, self.dropoff_loc]]
        self.agent_positions = {i: pos for i, pos in enumerate(random.sample(possible_positions, self.num_agents))}
        self.agent_is_carrying = {i: False for i in range(self.num_agents)}
        self._update_pos_lookup()
        return self.get_full_state()
    
    def _update_pos_lookup(self):
        self._pos_to_agent_id = {pos: agent_id for agent_id, pos in self.agent_positions.items()}

    def get_state_for_agent(self, agent_id):
        x, y = self.agent_positions[agent_id]
        neighbor_states = []
        for direction in Direction:
            dx, dy = direction.value
            check_pos = (x + dx, y + dy)
            if not (0 <= check_pos[0] < self.size and 0 <= check_pos[1] < self.size):
                neighbor_states.append(-1)
            elif check_pos in self._pos_to_agent_id:
                other_agent_id = self._pos_to_agent_id[check_pos]
                neighbor_states.append(2 if self.agent_is_carrying[other_agent_id] else 1)
            else:
                neighbor_states.append(0)
        target = self.dropoff_loc if self.agent_is_carrying[agent_id] else self.pickup_loc
        return (x, y, int(self.agent_is_carrying[agent_id]), target[0] - x, target[1] - y, *neighbor_states)
    
    def get_full_state(self):
        return {agent_id: self.get_state_for_agent(agent_id) for agent_id in range(self.num_agents)}

    def step(self, actions):
        rewards = {i: Reward.STEP_PENALTY.value for i in range(self.num_agents)}
        info = { LogKeys.DELIVERIES: 0, LogKeys.WALL_COLLISION: 0, LogKeys.HEAD_ON_COLLISION: 0 }
        
        new_positions = {}
        claimed_cells = set()
        sorted_agent_ids = sorted(actions.keys(), key=lambda id: (self.agent_is_carrying[id]), reverse=True)

        for agent_id in sorted_agent_ids:
            old_pos = self.agent_positions[agent_id]
            dx, dy = actions[agent_id].value
            intended_pos = (old_pos[0] + dx, old_pos[1] + dy)
            is_wall = not (0 <= intended_pos[0] < self.size and 0 <= intended_pos[1] < self.size)
            is_occupied = intended_pos in claimed_cells

            if not is_wall and not is_occupied:
                new_positions[agent_id] = intended_pos
                claimed_cells.add(intended_pos)
            else:
                new_positions[agent_id] = old_pos
                claimed_cells.add(old_pos)
                if is_wall:
                    rewards[agent_id] += Reward.WALL_COLLISION.value
                    info[LogKeys.WALL_COLLISION] += 1
                else:
                    rewards[agent_id] += Reward.YIELD_PENALTY.value
        
        self._old_positions = self.agent_positions.copy()
        self.agent_positions = new_positions
        self._update_pos_lookup()

        for agent_id in range(self.num_agents):
            pos, old_pos, carrying = self.agent_positions[agent_id], self._old_positions[agent_id], self.agent_is_carrying[agent_id]
            target = self.dropoff_loc if carrying else self.pickup_loc
            if (abs(pos[0] - target[0]) + abs(pos[1] - target[1])) < (abs(old_pos[0] - target[0]) + abs(old_pos[1] - target[1])):
                rewards[agent_id] += Reward.PROGRESS_REWARD.value
            if carrying and pos == self.dropoff_loc:
                self.agent_is_carrying[agent_id] = False
                rewards[agent_id] += Reward.SUCCESSFUL_DELIVERY.value
                info[LogKeys.DELIVERIES] += 1
            elif not carrying and pos == self.pickup_loc:
                self.agent_is_carrying[agent_id] = True
                rewards[agent_id] += Reward.PICKUP_INCENTIVE.value
                
        return self.get_full_state(), rewards, info