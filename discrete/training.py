import gymnasium as gym
from stable_baselines3 import PPO # wanna use this algo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os # creating folders and such

def linear_schedule(initial_value):
    # decaying from initial value to 0
    def schedule_func(progress_remaining):
        # ppo and most sb3 algos expect lr to either be a number
        # or a callable that takes progress_remaining as input
        # this is why we use progress_remaining
        return progress_remaining * initial_value
    return schedule_func

def train(seed=0):
    train_env = make_vec_env("LunarLander-v3", n_envs=4, seed=seed)
    train_env = VecNormalize(train_env) # normalization for making training more stable

    eval_env = DummyVecEnv([lambda: Monitor(gym.make("LunarLander-v3", render_mode=None))])
    eval_env = VecNormalize(eval_env, training=False)
    
    train_count = 3000000 # 3M

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    stop_training = StopTrainingOnRewardThreshold(reward_threshold=220, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        log_path="logs",
        best_model_save_path="models",
        # lunarlander is noisy so we need more eps per eval
        eval_freq=2500,
        n_eval_episodes=10,
        callback_on_new_best=stop_training
    )

    model = PPO(
        "MlpPolicy", # CnnPolicy for images, MlpPolicies for other types
        train_env,
        learning_rate=linear_schedule(3e-4),  # linear decay from 3e-4 to 0
        batch_size=64, #smaller batches do better in this env
        n_steps=2048, # steps before model trains itself
        n_epochs=10, # how many times model trains itself from step data
        gamma=0.999, # used calculating future vs immediate rewards
        gae_lambda=0.95,
        ent_coef= 0.01, # encouraging exploration
        max_grad_norm=0.5,
        seed=seed
)

    model.learn(
        total_timesteps=train_count,
        callback=eval_callback,
        progress_bar=True
    )

    train_env.save("models/vec_normalize.pkl")
    model.save("models/final_model") # saving final model

    return model, train_env


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
