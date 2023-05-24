from stable_baselines3 import PPO
from khunpanenv import KhunPanEnv
import cProfile

NUM_ITERATIONS = 100
NUM_STEPS = 100

env = KhunPanEnv()


def sample_run():
    for i in range(NUM_ITERATIONS):
        env.reset()
        for ep in range(NUM_STEPS):
            env.step(action)


model = PPO('MlpPolicy', env, verbose=1)
action = model.action_space.sample()
cProfile.run('sample_run()')
