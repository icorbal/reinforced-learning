from typing import Callable
from stable_baselines3.common.vec_env import SubprocVecEnv
from env import Env
from custom_logs import CustomTensorboardCallback
from stable_baselines3 import PPO, DQN
import time
import os


NUM_STEPS = 1000000
NUM_ENVS = 1

if __name__ == "__main__":
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)


    def make_env() -> Callable:
        def _init() -> Env:
            return Env()
        return _init


    if NUM_ENVS > 1:
        env = SubprocVecEnv([make_env() for i in range(NUM_ENVS)])
    else:
        env = Env()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cuda", learning_rate=0.0005)
    modelName = model.__class__.__name__
    iters = 0
    while True:
        iters += 1
        model.learn(
            total_timesteps=NUM_STEPS,
            reset_num_timesteps=False,
            tb_log_name=f"{modelName}-lr1e-5",
            callback=CustomTensorboardCallback(),
            progress_bar=True
        )
        model.save(f"{models_dir}/{NUM_STEPS * iters}")

