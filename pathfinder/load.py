from mayavi import mlab
import time
from threading import Thread

from sb3_contrib import QRDQN, TRPO
from env import Env
import os
from pathlib import Path
from stable_baselines3 import PPO, DQN, DDPG

from render_controller import RenderController

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

class ThreadedAction(Thread):
    def __init__(self, environment, **kwargs):
        Thread.__init__(self, **kwargs)
        self.env = environment

    def run(self):
        model = PPO.load(model_path, env=self.env)
        for ep in range(episodes):
            obs, info = self.env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, truncate, info = self.env.step(action)
                self.env.render()
                time.sleep(1/SPEED)


env = Env(render_controller=RenderController())
ThreadedAction(env).start()

env.start()
