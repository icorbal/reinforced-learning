from stable_baselines3 import PPO
from env import Env

env = Env()
env.reset()
model = PPO('MlpPolicy', env, verbose=1)
episodes = 10000

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        # pass observation to model to get predicted action
        action = model.action_space.sample()

        # pass action to env and get info back
        obs, rewards, done, info = env.step(action)

        # show the environment on the screen
        env.render()
