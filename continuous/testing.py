import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

def test_agent(model_path="models/best_model.zip", vec_normalize_path = "models/vec_normalize.pkl", episodes=5):
    env = gym.make("LunarLander-v3", render_mode="human", continuous=True)
    env = DummyVecEnv([lambda: env])

    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.norm_reward = False # disable reward normalization for more accuracy
        env.training = False # set to eval mode
    else:
        raise Exception("vecnormalize file not found")
    
    model = PPO.load(model_path, env=env) # linking model to env

    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=episodes,
                                       deterministic=True, render=True,
                                       return_episode_rewards=True)
    for i in range(episodes):
        print(f"episode:{i+1}, reward:{rewards[i]}, length:{lengths[i]}")

    env.close()

if __name__ == "__main__":
    test_agent()
