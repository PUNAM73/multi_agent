from train import run_training
from evaluate import run_evaluation

if __name__ == "__main__":
    # 1. Run the training process
    collisions_from_training = run_training()

    # 2. Run the evaluation process
    run_evaluation(training_collisions=collisions_from_training)