import gym
from stable_baselines3 import PPO
import time
import os
from humanoid_stand_env import HumanoidStandupEnv


models_dir = f"models/{int(time.time())}/"

logdir = f"logs/{int(time.time())}/"

#models_dir = f"models/01/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = HumanoidStandupEnv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

#model = PPO.load(f"{models_dir}/0", env=env)

TIMESTEPS = 1000000
iters = 0
#iters = 479
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
