from stable_baselines3 import PPO
from khunpanenv import KhunPanEnv

env = KhunPanEnv()

model = PPO('MlpPolicy', env, verbose=1)

for ep in range(1000):
    done = False
    while not done:
        action = model.action_space.sample()
        obs, rewards, done, info = env.step(action)
        env.render()
