from pathlib import Path

import gymnasium as gym
import os
from stable_baselines3 import PPO

paths = sorted(Path("models/").iterdir(), key=os.path.getmtime)
models_dir = paths[-1]

env = gym.make('HumanoidStandup-v4', render_mode='human')
env.reset()

dirpath = f"{models_dir}/"
paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)

model_path = f"{os.path.splitext(paths[-1])[0]}"
#model_path = "models/1646638573/100000"
model = PPO.load(model_path, env=env)

episodes = 100

vec_env = model.get_env()

for ep in range(episodes):
    obs = vec_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        #print(action)
        obs, rewards, done, info = vec_env.step(action)
        vec_env.render()
