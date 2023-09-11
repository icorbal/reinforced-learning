import gymnasium as gym
from gymnasium import spaces
import numpy as np
from maze import Maze
import math
import sys

MAX_STEPS = 200
N_CHANNELS = 1
HEIGHT = 50
WIDTH = 50

class Env(gym.Env):

    def __init__(self):
        super(Env, self).__init__()
        self.maze = Maze(WIDTH, HEIGHT)
        self.maze.generate()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        self.done = False
        self.totalRuns = 0
        self.numSuccesful = 0
        self.minDist = HEIGHT * WIDTH
        self.numSteps = 0
        self.rewardSteps = 0
        self.invalidSteps = 0
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=50, shape=(HEIGHT * WIDTH + 6,), dtype=np.uint8)
        #self.observation_space = spaces.Box(low=0, high=50, shape=(6,), dtype=np.uint8)
        self.obs = self.maze.getObs()

    def step(self, action):
        self.numSteps += 1
        oldPos = self.maze.pos
        self.maze.move(action)
        invalidStep = False
        if oldPos == self.maze.pos:
            self.invalidSteps+=1
            invalidStep = True
        if self.maze.pos == self.maze.exit:
            self.reward = 1000
            self.done = True
            self.numSuccesful += 1
        else:
            dist = math.dist(self.maze.exit, self.maze.pos)
            if dist < self.minDist:
                self.minDist = dist
                self.reward = 1
                self.rewardSteps += 1
            else:
                if invalidStep:
                    self.reward = 0
                else:    
                    self.reward = 0.2
        if self.numSteps > MAX_STEPS:
            self.done = True
        #print("reward = " + str(self.reward))
        self.obs = self.maze.getObs()
        return self.obs, self.reward, self.done, False, {}


    def reset(self, **kwargs):
        self.maze.generate()
        self.minDist = HEIGHT * WIDTH
        self.done = False
        self.totalRuns += 1
        info = {
            "num_steps": self.numSteps,
            "invalid_steps": self.invalidSteps,
            "reward_steps": self.rewardSteps,
            "num_successful": self.numSuccesful,
            "totalRuns": self.totalRuns
        }
        self.numSteps = 0
        self.invalidSteps = 0
        self.rewardSteps = 0
        self.obs = self.maze.getObs()
        return self.obs, info

    def render(self):
        self.maze.render()
