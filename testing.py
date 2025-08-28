import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import os


def test_agent(model_path="models/best_model.zip", vec_normalize_path = "models/vec_normalize.pkl", episodes=5):
    env = gym.make("LunarLander-v3", render_mode="human")
    env = DummyVecEnv([lambda: env])

    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
    else:
        raise Exception("vecnormalize file not found")
    
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action) # different return format for vecenv
            done = done[0]
            total_reward += reward[0]

        print(f"episode {ep+1}, reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test_agent()
