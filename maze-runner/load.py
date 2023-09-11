import time
from env import Env
import os
from pathlib import Path
from stable_baselines3 import PPO, DQN, DDPG

SPEED = 100

paths = sorted(Path("models/").iterdir(), key=os.path.getmtime)
models_dir = paths[-1]

dirpath = f"{models_dir}/"
#dirpath = f"models/1685268675/"
paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)
model_path = f"{os.path.splitext(paths[-1])[0]}"
#model_path = "models/no_lvl_penalty.zip"
print(f"loading model: {model_path}")

episodes = 100

env = Env()

model = PPO.load(model_path, env=env)
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, truncate, info = env.step(action)
        env.render()
        time.sleep(1/SPEED)


