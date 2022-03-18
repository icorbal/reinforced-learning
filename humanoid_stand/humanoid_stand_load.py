import gym, os, time
from pathlib import Path
from stable_baselines3 import PPO

paths = sorted(Path("models/").iterdir(), key=os.path.getmtime)
models_dir = paths[-1]

env = gym.make('HumanoidStandup-v2')
env.reset()

dirpath = f"{models_dir}/"
paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)

model_path = f"{os.path.splitext(paths[-1])[0]}"
#model_path = "models/1646638573/100000"
model = PPO.load(model_path, env=env)

episodes = 100

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        #print(action)
        obs, rewards, done, info = env.step(action)
        env.render()
