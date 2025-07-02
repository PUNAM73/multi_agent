# train.py

import time
from collections import defaultdict
import config
from environment import GridWorld, LogKeys
from agent import SharedQPolicy
from plotter import plot_history

def run_training():
    print("--- Starting Training ---")
    shared_policy = SharedQPolicy()
    start_time = time.time()

    stages = [
        {'name': 'Stage 1: Basics', 'num_agents': 2, 'steps': int(config.FULL_TRAINING_STEPS * 0.4)},
        {'name': 'Stage 2: Intermediate', 'num_agents': 3, 'steps': int(config.FULL_TRAINING_STEPS * 0.3)},
        {'name': 'Stage 3: Full Complexity', 'num_agents': 4, 'steps': int(config.FULL_TRAINING_STEPS * 0.3)}
    ]

    global_step, total_deliveries, total_collisions, total_wall_collisions, total_agent_steps = 0, 0, 0, 0, 0
    history = defaultdict(list)

    for stage_info in stages:
        print(f"\n--- Entering {stage_info['name']} ({stage_info['num_agents']} agents) ---")
        env = GridWorld(num_agents=stage_info['num_agents'])
        states = env.reset()
        episode_deliveries = 0

        for stage_step in range(stage_info['steps']):
            if global_step >= config.FULL_TRAINING_STEPS or (time.time() - start_time) > config.WALLTIME_BUDGET:
                break
            
            global_step += 1
            total_agent_steps += env.num_agents

            actions = {i: shared_policy.choose_action(states[i]) for i in range(env.num_agents)}
            next_states, rewards, info = env.step(actions)
            for i in range(env.num_agents):
                shared_policy.update_q_table(states[i], actions[i], rewards[i], next_states[i])
            
            states = next_states
            shared_policy.decay_epsilon()

            total_deliveries += info.get(LogKeys.DELIVERIES, 0)
            total_wall_collisions += info.get(LogKeys.WALL_COLLISION, 0)
            episode_deliveries += info.get(LogKeys.DELIVERIES, 0)

            if global_step % config.EPISODE_LENGTH == 0:
                episode_num = global_step // config.EPISODE_LENGTH
                history[LogKeys.EPISODES].append(episode_num)
                history[LogKeys.EPISODE_DELIVERIES].append(episode_deliveries)
                history[LogKeys.EPSILON].append(shared_policy.epsilon)
                print(f"Step {global_step}/{config.FULL_TRAINING_STEPS} | Ep {episode_num} | Deliveries: {episode_deliveries} | Epsilon: {shared_policy.epsilon:.4f}")
                episode_deliveries = 0
        
        if global_step >= config.FULL_TRAINING_STEPS or (time.time() - start_time) > config.WALLTIME_BUDGET:
            print("Training budget exceeded. Stopping.")
            break

    final_time = time.time() - start_time
    avg_steps_per_delivery = (total_agent_steps / total_deliveries) if total_deliveries > 0 else float('inf')

    print(f"\n--- Training Finished in {final_time:.2f}s ---")
    print("\n--- Final Training Report ---")
    print(f"Total Global Steps Run:       {global_step}")
    print(f"Total Agent Steps:            {total_agent_steps}")
    print(f"Total Deliveries:             {total_deliveries}")
    print(f"Total Collisions:             {total_collisions}")
    print(f"Wall Collisions:              {total_wall_collisions}")
    print(f"Avg Agent Steps/Delivery:     {avg_steps_per_delivery:.2f}")

    shared_policy.save_q_table()
    plot_history(history)
    
    return total_collisions # Return this for the evaluation script