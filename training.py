import gymnasium as gym
from stable_baselines3 import PPO # wanna use this algo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import linear_schedule
import os # creating folders and such

def train(seed=0):
    train_env = make_vec_env("LunarLander-v3", n_envs=4, seed=seed)
    eval_env = Monitor(gym.make("LunarLander-v3", render_mode="human"))
    train_count = 3000000 # 3M

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        log_path="logs",
        best_model_save_path="models",
        eval_freq=1000
    )

    model = PPO(
        "MlpPolicy", # CnnPolicy for images, MlpPolicies for other types
        train_env,
        learning_rate=linear_schedule(3e-4, 1e-5),  # linear decay from 3e-4 to 1e-5
        batch_size=128,
        n_steps=2048, # steps before model trains itself
        n_epochs=10, # how many times model trains itself from step data
        gamma=0.99, # used calculating future vs immediate rewards
        gae_lambda=0.95,
        seed=seed
)

    model.learn(
        total_timesteps=train_count,
        callback=eval_callback,
        progress_bar=True
    )


def run_multi_seed_training(seeds=[0, 42, 123, 456, 789]):
    # multiple seeds for more robust training, more scientific but needs more time
    for seed in seeds:
        print(f"training with seed of {seed}")
        train(seed)

if __name__ == "__main__":
    # single seed training
    train()
    
    # multi-seed training
    # run_multi_seed_training()
