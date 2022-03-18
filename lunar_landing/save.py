import gym
from stable_baselines3 import PPO,DQN
from snakeenv import SnekEnv
import os


models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('LunarLander-v2')
env.reset()

model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN-1")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
