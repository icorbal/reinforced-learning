import os
from pathlib import Path
from sb3_contrib import TRPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from typing import Callable

from stable_baselines3.common.vec_env import SubprocVecEnv

from custom_logs import CustomTensorboardCallback
from env import Env

NUM_ENVS = 1 # Number of processes to use
TIMESTEPS = 1000000 # Number of timesteps to train for

if __name__ == "__main__":
    model_paths = sorted(Path("./models/").iterdir(), key=os.path.getmtime)
    models_dir = model_paths[-1]

    training_paths = sorted(Path(models_dir).iterdir(), key=os.path.getmtime)
    training_file = training_paths[-1]

    log_paths = sorted(Path("./logs/").iterdir(), key=os.path.getmtime)
    logdir = log_paths[-1]

    def make_env() -> Callable:
        def _init() -> Env:
            return Env()
        return _init

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if NUM_ENVS > 1:
        env = SubprocVecEnv([make_env() for i in range(NUM_ENVS)])
    else:
        env = Env()

    training_path = f"{os.path.splitext(training_file)[0]}"
    print(f"loading model: {training_path}")
    model = PPO.load(training_path, env=env, device="cuda", n_cpu=NUM_ENVS)
    modelName = model.__class__.__name__

    iters = int(training_path.split(os.path.sep)[2])
    while True:
        iters += 1
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=f"{modelName}10k",
            callback=CustomTensorboardCallback()
        )
        model.save(f"{models_dir}/{TIMESTEPS * iters}")
