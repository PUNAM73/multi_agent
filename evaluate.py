import config
from environment import GridWorld, LogKeys, Direction
from agent import SharedQPolicy
import numpy as np

def run_evaluation(training_collisions=0):
    print("\n--- Starting Evaluation ---")
    env = GridWorld(num_agents=config.NUM_AGENTS)
    
    # Load the pre-trained policy
    policy = SharedQPolicy()
    try:
        policy.load_q_table()
    except FileNotFoundError:
        print(f"ERROR: Could not find the Q-table file '{config.Q_TABLE_FILE_NAME}'.")
        print("Please run the training script first to generate the file.")
        return # Exit if the model doesn't exist
    
    policy.epsilon = 0.0  # Turn off exploration for evaluation


    # Add counters for the new metrics
    total_evaluation_collisions = 0
    collision_free_scenarios = 0
    all_successful_delivery_steps = []
    total_successful_deliveries = 0
    total_failed_deliveries = 0

    LOG_INTERVAL = 1 
    print(f"Evaluating {config.EVALUATION_EPISODES} scenarios. Logging every {LOG_INTERVAL} scenarios.")

    for scenario_num in range(1, config.EVALUATION_EPISODES + 1):
        states = env.reset()
        agents_with_items = {i for i, carrying in env.agent_is_carrying.items() if carrying}
        
        # Track collisions and delivery start times for this specific scenario
        scenario_had_collision = False
        scenario_collisions = 0
        scenario_deliveries = 0
        scenario_delivery_steps = []
        delivery_start_steps = {}

        for step in range(config.EVALUATION_MAX_STEPS):
            actions = {i: policy.choose_action(states[i]) for i in range(env.num_agents)}
            states, _, info = env.step(actions)

            # Track collisions for this scenario
            step_collisions = info.get(LogKeys.HEAD_ON_COLLISION, 0)
            if step_collisions > 0:
                scenario_collisions += step_collisions
                scenario_had_collision = True

            # Track deliveries for this scenario
            delivered_this_step = {i for i in agents_with_items if not env.agent_is_carrying[i]}
            if delivered_this_step:
                scenario_deliveries += len(delivered_this_step)
                for agent_id in delivered_this_step:
                    if agent_id in delivery_start_steps:
                        duration = step - delivery_start_steps[agent_id]
                        scenario_delivery_steps.append(duration)
                        del delivery_start_steps[agent_id]
                agents_with_items -= delivered_this_step

            newly_carrying = {i for i, carrying in env.agent_is_carrying.items() if carrying and i not in agents_with_items}
            agents_with_items.update(newly_carrying)
            if newly_carrying:
                 for agent_id in newly_carrying:
                     # Record the start time of the delivery attempt
                     delivery_start_steps[agent_id] = step
            agents_with_items.update(newly_carrying)
        
        # --- Update Cumulative Counters After Scenario Ends ---
        total_evaluation_collisions += scenario_collisions
        if not scenario_had_collision:
            collision_free_scenarios += 1

        total_successful_deliveries += scenario_deliveries
        total_failed_deliveries += len(agents_with_items)
        all_successful_delivery_steps.extend(scenario_delivery_steps)

        # The per-scenario logging block
        if scenario_num % LOG_INTERVAL == 0:
            avg_steps = np.mean(scenario_delivery_steps) if scenario_delivery_steps else 0
            # MODIFIED PRINT STATEMENT
            print(f"Scenario {scenario_num}/{config.EVALUATION_EPISODES} | " f"Deliveries: {scenario_deliveries} | " f"Collisions: {scenario_collisions} | " f"Avg Steps: {avg_steps:.2f}")

    # --- Calculate Final Metrics ---
    total_attempts = total_successful_deliveries + total_failed_deliveries
    success_rate = (total_successful_deliveries / total_attempts) * 100 if total_attempts > 0 else 0
    
    # Calculate collision percentage and average steps
    collision_free_percentage = (collision_free_scenarios / config.EVALUATION_EPISODES) * 100
    avg_steps_per_delivery = np.mean(all_successful_delivery_steps) if all_successful_delivery_steps else float('inf')
    
    # Calculate performance points (B)
    performance_points_b = 0
    if success_rate > 95:
        performance_points_b = 2
    elif success_rate > 85:
        performance_points_b = 1

    alpha = 1 - (33 / 200) * max(0, config.OPTION_COST_C - performance_points_b)

    print("\n--- Final Evaluation Report ---")
    print(f"Total Scenarios Run:              {config.EVALUATION_EPISODES}")
    print(f"Delivery Success Rate:            {success_rate:.2f}% ({total_successful_deliveries}/{total_attempts} attempts)")
    print(f"Average Steps per Delivery:       {avg_steps_per_delivery:.2f}")
    print("-" * 35)
    print(f"Total Collisions During Training: {training_collisions}")
    print(f"Total Collisions During Evaluation: {total_evaluation_collisions}")
    print(f"Collision-Free Scenarios:         {collision_free_percentage:.2f}% ({collision_free_scenarios}/{config.EVALUATION_EPISODES})")
    print("-" * 35)
    print(f"Option Cost (C):                  {config.OPTION_COST_C}")
    print(f"Performance Points (B):           {performance_points_b}")
    print(f"Scaling Factor (a):               {alpha:.4f}")
    
    if success_rate / 100 >= config.MIN_SUCCESS_RATE:
        print(f"STATUS: PASSED (Success rate >= {config.MIN_SUCCESS_RATE * 100}%)")
    else:
        print(f"STATUS: FAILED (Success rate < {config.MIN_SUCCESS_RATE * 100}%)")
