from sb3_contrib import QRDQN, TRPO

from khunpanenv import KhunPanEnv
import os
from pathlib import Path
from stable_baselines3 import PPO, DQN, DDPG

paths = sorted(Path("models/").iterdir(), key=os.path.getmtime)
models_dir = paths[-1]

env = KhunPanEnv()

dirpath = f"{models_dir}/"
paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)

model_path = f"{os.path.splitext(paths[-1])[0]}"
model = TRPO.load(model_path, env=env)

episodes = 100


for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncate, info = env.step(action)
        env.render()
