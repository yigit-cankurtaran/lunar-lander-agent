import gymnasium as gym
from stable_baselines3 import PPO # wanna use this algo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os # creating folders and such

def train():
    train_env = make_vec_env("LunarLander-v3", n_envs=4, seed=0)
    eval_env = Monitor(gym.make("LunarLander-v3"))
    train_count = 1000000 # 1M

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        log_path="logs",
        best_model_save_path="models",
        eval_freq=1000
    )

    model = PPO(
        "MlpPolicy"
    )
    


if __name__ == "__main__":
    train()
