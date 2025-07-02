import config
from environment import GridWorld
from agent import SharedQPolicy

def run_evaluation(training_collisions=0):
    print("\n--- Starting Evaluation ---")
    env = GridWorld(num_agents=config.NUM_AGENTS)
    
    # Load the pre-trained policy
    policy = SharedQPolicy()
    policy.load_q_table()
    policy.epsilon = 0.0  # Turn off exploration for evaluation

    successful_deliveries, failed_deliveries = 0, 0

    for _ in range(config.EVALUATION_EPISODES):
        states = env.reset()
        agents_with_items = {i for i, carrying in env.agent_is_carrying.items() if carrying}

        for _ in range(config.EVALUATION_MAX_STEPS):
            actions = {i: policy.choose_action(states[i]) for i in range(env.num_agents)}
            states, _, info = env.step(actions)
            delivered_this_step = {i for i in agents_with_items if not env.agent_is_carrying[i]}
            if delivered_this_step:
                successful_deliveries += len(delivered_this_step)
                agents_with_items -= delivered_this_step
            newly_carrying = {i for i, carrying in env.agent_is_carrying.items() if carrying and i not in agents_with_items}
            agents_with_items.update(newly_carrying)
        
        failed_deliveries += len(agents_with_items)

    total_attempts = successful_deliveries + failed_deliveries
    success_rate = (successful_deliveries / total_attempts) * 100 if total_attempts > 0 else 0
    
    # Calculate performance points (B)
    performance_points_b = 0
    if success_rate > 95:
        performance_points_b = 2
    elif success_rate > 85:
        performance_points_b = 1

    alpha = 1 - (33 / 200) * max(0, config.OPTION_COST_C - performance_points_b)

    print("\n--- Final Evaluation Report ---")
    print(f"Total Scenarios Run: {config.EVALUATION_EPISODES}")
    print(f"Success Rate: {success_rate:.3f}% ({successful_deliveries}/{total_attempts})")
    print(f"Total Collisions During Training: {training_collisions}")
    print(f"Option Cost (C): {config.OPTION_COST_C}")
    print(f"Performance Points (B): {performance_points_b}")
    print(f"Scaling Factor (a): {alpha:.4f}")
    if success_rate / 100 >= config.MIN_SUCCESS_RATE:
        print(f"STATUS: PASSED (Success rate >= {config.MIN_SUCCESS_RATE * 100}%)")
    else:
        print(f"STATUS: FAILED (Success rate < {config.MIN_SUCCESS_RATE * 100}%)")