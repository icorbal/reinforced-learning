import gym
from stable_baselines3 import PPO
import time
import os
from humanoid_stand_env import HumanoidStandupEnv
from pathlib import Path

model_paths = sorted(Path("models/").iterdir(), key=os.path.getmtime)
models_dir = model_paths[-1]

training_paths = sorted(Path(models_dir).iterdir(), key=os.path.getmtime)
training_file = training_paths[-1]

log_paths = sorted(Path("logs/").iterdir(), key=os.path.getmtime)
logdir = log_paths[-1]


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = HumanoidStandupEnv()
env.reset()

training_path = f"{os.path.splitext(training_file)[0]}"
model = PPO.load(training_path, env=env)

TIMESTEPS = 100000
iters = int(training_path.rpartition("/")[2])
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
