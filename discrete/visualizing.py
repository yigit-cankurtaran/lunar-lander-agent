import numpy as np
import matplotlib.pyplot as plt
import os

def graph():
    if not os.path.exists("logs/evaluations.npz"):
        return Exception("no log file, run training")
    
    data = np.load("logs/evaluations.npz")
    timesteps = data['timesteps']
    rewards = data['results']
    
    mean_rewards = np.mean(rewards, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Training Rewards Over Time')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    graph()
