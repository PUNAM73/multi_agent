import matplotlib.pyplot as plt
from environment import LogKeys

def plot_history(history):
    if not history:
        print("No history to plot.")
        return

    print("\n--- Plotting Metrics ---")
    plt.figure(figsize=(12, 6))

    # Plot 1: Deliveries
    plt.subplot(1, 2, 1)
    plt.plot(history[LogKeys.EPISODES], history[LogKeys.EPISODE_DELIVERIES], color='green')
    plt.title('Deliveries per Logged Episode')
    plt.xlabel('Episode Number')
    plt.ylabel('Deliveries')
    plt.grid(True)

    # Plot 2: Epsilon Decay
    plt.subplot(1, 2, 2)
    plt.plot(history[LogKeys.EPISODES], history[LogKeys.EPSILON], color='orange')
    plt.title('Epsilon Decay Over Time')
    plt.xlabel('Episode Number')
    plt.ylabel('Epsilon')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_summary.png")
    print("Saved training summary plot to training_summary.png")
    plt.show()