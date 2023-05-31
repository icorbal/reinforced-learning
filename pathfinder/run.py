import time
from threading import Thread
from mayavi import mlab
from stable_baselines3 import PPO
from env import Env, UP, DOWN, LEFT, RIGHT
from pynput import keyboard


class ThreadedAction(Thread):

    def __init__(self, env, **kwargs):
        Thread.__init__(self, **kwargs)
        self.env = env
        self.action = 0  # model.action_space.sample()


    def run(self):
        model = PPO('MlpPolicy', self.env, verbose=1)
        for ep in range(1000):
            obs, info = self.env.reset()
            done = False
            while not done:
                obs, rewards, done, truncate, info = self.env.step(self.action)
                self.env.render()
                time.sleep(0.1)

    def on_press(self, key):
        if key == keyboard.Key.up:
            self.action = UP
        elif key == keyboard.Key.down:
            self.action = DOWN
        elif key == keyboard.Key.left:
            self.action = LEFT
        elif key == keyboard.Key.right:
            self.action = RIGHT


env = Env(render="human")

ta = ThreadedAction(env)
ta.start()
listener = keyboard.Listener(on_press=ta.on_press)
listener.start()  # start to listen on a separate thread
#listener.join()  # remove if main thread is polling self.keys

env.start()
