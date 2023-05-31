
from stable_baselines3 import PPO
import time
import os
from typing import Callable
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
from humanoidstandup_v4 import HumanoidStandupEnv
from custom_logs import CustomTensorboardCallback

NUM_STEPS = 1000000
NUM_ENVS = 4


def learn():
	model.learn(
		total_timesteps=NUM_STEPS,
		reset_num_timesteps=False,
		tb_log_name=f"{modelName}1M",
		callback=CustomTensorboardCallback(),
		progress_bar=True
	)


if __name__ == "__main__":
	models_dir = f"models/{int(time.time())}/"
	logdir = f"logs/{int(time.time())}/"

	if not os.path.exists(models_dir):
		os.makedirs(models_dir)

	if not os.path.exists(logdir):
		os.makedirs(logdir)

	def make_env() -> Callable:
		def _init() -> gym.Env:
			return HumanoidStandupEnv()
		return _init

	if NUM_ENVS > 1:
		env = SubprocVecEnv([make_env() for i in range(NUM_ENVS)])
	else:
		env = HumanoidStandupEnv()

	model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=logdir)
	modelName = model.__class__.__name__
	iters = 0
	while True:
		iters += 1
		learn()
		model.save(f"{models_dir}/{NUM_STEPS * iters}")
