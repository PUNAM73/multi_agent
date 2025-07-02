# --- Training & Environment Constants ---
FULL_TRAINING_STEPS = 1_500_000
EPISODE_LENGTH = 2500  # For logging purposes
TRAINING_COLLISION_BUDGET = 4000 # Increased for full training run
WALLTIME_BUDGET = 10 * 60  # 10 minutes
GRID_SIZE = 5
NUM_AGENTS = 4

# --- Evaluation Constants ---
EVALUATION_EPISODES = 300
EVALUATION_MAX_STEPS = 350
MIN_SUCCESS_RATE = 0.75

# --- File & Model Constants ---
Q_TABLE_FILE_NAME = "shared_q_table.json"

# --- Option Costs for Final Grade Calculation ---
SENSORS_COST = 2
CENTRAL_CLOCK_COST = 1
STAGED_TRAINING_COST = 3
OPTION_COST_C = SENSORS_COST + CENTRAL_CLOCK_COST + STAGED_TRAINING_COST

# --- Agent Hyperparameters ---
ALPHA = 0.2         # Learning Rate
GAMMA = 0.99        # Discount Factor
EPSILON = 1.0       # Initial Exploration Rate
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.1