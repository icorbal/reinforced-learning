import gym, os
from pathlib import Path
from stable_baselines3 import PPO
from snakeenv import SnekEnv

paths = sorted(Path("models/").iterdir(), key=os.path.getmtime)
models_dir = paths[-1]
#models_dir = "models/1644878689"

env = SnekEnv()
env.reset()

dirpath = f"{models_dir}/"
paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)

#model_path = f"{models_dir}/18600000"
#print(paths[-1])
model_path = f"{os.path.splitext(paths[-1])[0]}"
model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        # print(rewards)
