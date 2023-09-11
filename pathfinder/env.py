import math
import time

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from reward_calculator import RewardCalculator
from world import World, calculate_next_point
from world import LEFT, RIGHT, UP, DOWN, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT

MAX_ITER = 1000
OBS_DIM = 40
MAX_LEVEL_DIFF = 0.001
LEVEL_PENALTY = -1
NUM_OBS = 8

action_array = [DOWN_LEFT, DOWN, DOWN_RIGHT, RIGHT, UP_RIGHT, UP, UP_LEFT, LEFT]


def random_point():
    return np.random.randint(50, 950), np.random.randint(50, 950)


class Env(gym.Env):

    def __init__(self, render_controller=None):
        super(Env, self).__init__()
        self.reward = None
        self.render_controller = render_controller
        self.observations = np.zeros((NUM_OBS,), dtype='uint8')
        self.done = False
        self.neutral_moves = self.num_steps = self.positive_moves = self.negative_moves = 0
        self.current_point = random_point()
        self.destination_point = random_point()
        self.initial_dist = self.distance_to_dest()
        self.world = World(MAX_LEVEL_DIFF, OBS_DIM)
        if self.render_controller is not None:
            self.render_controller.set_world(self.world)
        self.current_level = self.calc_current_level()
        self.action_space = spaces.Discrete(8)
        """
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, OBS_DIM, OBS_DIM),
            dtype=np.uint8
        )
        """
        """
        self.observation_space = spaces.Box(
            low=-5, high=5,
            shape=(NUM_OBS,),
            dtype=np.float32
        )
        """
        self.observation_space = spaces.Box(
            low=-1000, high=1000,
            shape=(5,),
            dtype=np.float32
        )
        self.reward_calculator = RewardCalculator(self.world, self.initial_dist, self.destination_point, MAX_LEVEL_DIFF)
        self.reward_calculator.update_current_point(self.current_point)

    def step(self, action):
        # self.check_predictions()
        # print(self.observations)
        next_point = calculate_next_point(self.current_point, action)
        self.reward = self.reward_calculator.calculate_reward(next_point)
        if self.reward < 0:
            self.negative_moves += 1
        elif self.reward > 0:
            self.positive_moves += 1
        else:
            self.neutral_moves += 1
        if 0 <= next_point[0] < 1000 and 0 <= next_point[1] < 1000:
            self.current_point = next_point
            self.world.walked_points.append(self.current_point)
            if self.render_controller is not None:
                self.render_controller.add_point(self.current_point)
        self.num_steps += 1
        if self.current_point == self.destination_point:
            self.reward = 1000
            # print("FINISHED!")
        self.done = self.num_steps >= MAX_ITER or self.current_point == self.destination_point
        self.reward_calculator.update_current_point(self.current_point)
        self.observations = self.get_observations()
        """
        dist = self.distance_to_dest()
        if dist < 50:
            self.display_image(self.observations[0])
            time.sleep(0.2)
            #print(self.reward)
            #print(dist)
        """
        """
        self.display_image(self.observations[0])
        time.sleep(0.2)
        print("Reward: ", self.reward, " Distance: ", self.distance_to_dest(), " current_point: ", self.current_point, " destination_point: ", self.destination_point, "action: ", action)
        """
        return self.observations, self.reward, self.done, False, {}

    def reset(self, **kwargs):
        self.done = False
        self.current_point = random_point()
        self.current_level = self.calc_current_level()
        self.destination_point = random_point()
        info = {
            "num_steps": self.num_steps,
            "neutral_moves": self.neutral_moves,
            "positive_moves": self.positive_moves,
            "negative_moves": self.negative_moves
        }
        self.world.reset(self.destination_point)
        self.initial_dist = self.distance_to_dest()
        self.reward_calculator = RewardCalculator(self.world, self.initial_dist, self.destination_point, MAX_LEVEL_DIFF)
        self.reward_calculator.update_current_point(self.current_point)
        self.neutral_moves = self.num_steps = self.positive_moves = self.negative_moves = 0
        if self.render_controller is not None:
            self.render_controller.reset()
            self.render_controller.set_start_point(self.current_point)
            self.render_controller.set_dest_point(self.destination_point)
        self.observations = self.get_observations()
        return self.observations, info

    def display_image(self, image_array):
        cv2.imshow('Image', image_array)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

    def render(self):
        pass

    def start(self):
        if self.render_controller is not None:
            self.render_controller.start()

    def get_observations(self):
        """
        return self.world.take_snapshot(self.current_point, self.destination_point)
        return self.world.get_levels(
            EXPLORATION_RADIUS,
            self.current_point[0],
            self.current_point[1],
            self.reward_calculator
        )
        """
        return [
            self.current_point[0],
            self.current_point[1],
            self.destination_point[0] - self.current_point[0],
            self.destination_point[1] - self.current_point[1],
            self.distance_to_dest()
        ]

    def distance_to_dest(self):
        return math.dist(self.destination_point, self.current_point)

    def calculate_lvl_pred(self, action_index):
        return self.observations[action_index]

    def calc_current_level(self):
        return self.world.calc_level(self.current_point)
