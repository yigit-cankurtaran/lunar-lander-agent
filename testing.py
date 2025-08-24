import gymnasium as gym
from stable_baselines3 import PPO

def test_agent(model_path="models/best_model.zip", episodes=5):
    env = gym.make("LunarLander-v3", render_mode="human")
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"episode {ep+1}, reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test_agent()
