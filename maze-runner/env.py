import gymnasium
from gymnasium import spaces
import numpy as np
from maze import Maze
import math

MAX_STEPS = 100
N_CHANNELS = 1
HEIGHT = 40
WIDTH = 40


class Env(gymnasium.Env):

    def __init__(self):
        super(Env, self).__init__()
        self.reward = None
        self.maze = Maze(WIDTH, HEIGHT)
        self.maze.generate()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        self.done = False
        self.totalRuns = 0
        self.numSuccessful = 0
        self.minDist = HEIGHT * WIDTH
        self.numSteps = 0
        self.rewardSteps = 0
        self.invalidSteps = 0
        self.prev_steps = set()
        self.info = {}
        self.total_reward = 0
        # Example for using image as input (channel-first; channel-last also works):
        #self.observation_space = spaces.Box(low=0, high=2, shape=(HEIGHT * WIDTH,), dtype=np.uint8)
        #self.observation_space = spaces.Box(low=0, high=255, shape=(1, HEIGHT, WIDTH), dtype=np.uint8)
        #self.observation_space = spaces.Box(low=0, high=50, shape=(6,), dtype=np.uint8)
        self.observation_space = spaces.Dict({
                "img": spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8),
                "info": spaces.Box(low=0, high=100, shape=(9,), dtype=np.uint8)
                #"discrete": spaces.Discrete(4),
            }
        )
        self.obs = self.get_obs(0)

    def get_obs(self, action):
        return {
            "img": self.maze.getImg(),
            "info": np.append([
                self.maze.pos[0], self.maze.pos[1],
                self.maze.exit[0], self.maze.exit[1],
                self.get_dist()
            ], self.maze.nextPos())
        }

    def get_dist(self):
        return round(abs(math.dist(self.maze.exit, self.maze.pos)))

    def step(self, action):
        self.numSteps += 1
        old_pos = self.maze.pos
        self.maze.move(action)
        self.reward = 0
        if old_pos == self.maze.pos:
            self.invalidSteps += 1
            self.reward = -1
        elif self.maze.pos == self.maze.exit:
            self.reward = 1000
            self.done = True
            self.numSuccessful += 1
        else:
            dist = abs(math.dist(self.maze.exit, self.maze.pos))
            if dist < self.minDist:
                self.minDist = dist
                self.reward = 1
                self.rewardSteps += 1
            else:
                if self.maze.pos in self.prev_steps:
                    self.reward = -0.2
                else:
                    self.reward = 0.2
        self.prev_steps.add(self.maze.pos)
        if self.numSteps > MAX_STEPS:
            self.done = True
        # print("reward = " + str(self.reward))
        self.obs = self.get_obs(action)
        self.total_reward += self.reward
        return self.obs, self.reward, self.done, False, {}

    def reset(self, **kwargs):
        self.maze.generate()
        self.minDist = HEIGHT * WIDTH
        self.done = False
        self.totalRuns += 1
        self.info = self.get_info()
        self.numSteps = 0
        self.invalidSteps = 0
        self.reward = 0
        self.total_reward = 0
        self.rewardSteps = 0
        self.prev_steps = set()
        self.obs = self.get_obs(0)
        return self.obs, self.info

    def get_info(self):
        return {
            "invalid_steps": self.invalidSteps,
            "reward_steps": self.rewardSteps,
            "total_reward": self.total_reward,
            "num_successful": self.numSuccessful,
            "totalRuns": self.totalRuns
        }

    def render(self):
        self.maze.render(self.get_info())
