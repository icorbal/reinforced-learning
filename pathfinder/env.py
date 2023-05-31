import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from reward_calculator import RewardCalculator
from world import World, calculate_next_point
from world import LEFT, RIGHT, UP, DOWN, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT

MAX_ITER = 4000
EXPLORATION_RADIUS = 1
MAX_LEVEL_DIFF = 0.0008
LEVEL_PENALTY = -1
NUM_OBS = 8

action_array = [DOWN_LEFT, DOWN, DOWN_RIGHT, RIGHT, UP_RIGHT, UP, UP_LEFT, LEFT]


def random_point():
    return 100 + np.random.randint(0, 900), 100 + np.random.randint(0, 900)


class Env(gym.Env):

    def __init__(self, render_controller=None):
        super(Env, self).__init__()
        self.reward = None
        self.render_controller = render_controller
        self.observations = np.zeros((NUM_OBS,), dtype='uint8')
        self.done = False
        self.level_penalty = 0
        self.out_of_boundaries = 0
        self.correct_moves = 0
        self.current_point = random_point()
        self.destination_point = random_point()
        self.initial_dist = self.distance_to_dest()
        self.world = World(MAX_LEVEL_DIFF)
        if self.render_controller is not None:
            self.render_controller.set_world(self.world)
        self.current_level = self.calc_current_level()
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(NUM_OBS,), dtype=np.float32)
        self.num_steps = 0
        self.reward_calculator = RewardCalculator(self.world, self.initial_dist, self.destination_point, MAX_LEVEL_DIFF)
        self.reward_calculator.update_current_point(self.current_point)

    def step(self, action):
        #self.check_predictions()
        #print(self.observations)
        next_point = calculate_next_point(self.current_point, action)
        if next_point[0] < 0 or next_point[0] >= 1000 or next_point[1] < 0 or next_point[1] >= 1000:
            self.reward = -5
            self.out_of_boundaries += 1
        else:
            self.reward = self.observations[action_array.index(action)]
            if self.reward == -5:
                self.level_penalty += 1
            else:
                self.correct_moves += 1
            self.current_point = next_point
            if self.render_controller is not None:
                self.render_controller.add_point(self.current_point)
        self.num_steps += 1
        if self.current_point == self.destination_point:
            self.reward = 10000
        self.done = self.num_steps >= MAX_ITER or self.current_point == self.destination_point
        self.reward_calculator.update_current_point(self.current_point)
        self.observations = self.get_observations()
        #print(self.reward)
        return self.observations, self.reward, self.done, False, {}

    def reset(self, **kwargs):
        self.done = False
        self.current_point = random_point()
        self.current_level = self.calc_current_level()
        self.destination_point = random_point()
        info = {
            "num_steps": self.num_steps,
            "level_penalty": self.level_penalty,
            "out_of_boundaries": self.out_of_boundaries,
            "correct_moves": self.correct_moves,
            "penalty/correct": self.level_penalty / self.correct_moves if self.correct_moves > 0 else 0
        }
        self.level_penalty = 0
        self.initial_dist = self.distance_to_dest()
        self.reward_calculator = RewardCalculator(self.world, self.initial_dist, self.destination_point, MAX_LEVEL_DIFF)
        self.reward_calculator.update_current_point(self.current_point)
        self.out_of_boundaries = 0
        self.correct_moves = 0
        self.num_steps = 0
        if self.render_controller is not None:
            self.render_controller.reset()
            self.render_controller.set_start_point(self.current_point)
            self.render_controller.set_dest_point(self.destination_point)
        self.observations = self.get_observations()
        return self.observations, info

    def render(self):
        pass

    def start(self):
        if self.render_controller is not None:
            self.render_controller.start()

    def get_observations(self):
        return self.world.get_levels(
            EXPLORATION_RADIUS,
            self.current_point[0],
            self.current_point[1],
            self.reward_calculator
        )

    def distance_to_dest(self):
        return math.dist(self.destination_point, self.current_point)

    def calculate_lvl_pred(self, action_index):
        return self.observations[action_index]

    def calc_current_level(self):
        return self.world.calc_level(self.current_point)

    def check_predictions(self):
        base_level = self.current_level
        fail = False
        fails = []
        diff_values = []
        values = []
        i = 0
        for action in action_array:
            point = self.calculate_next_point(action)
            if point[0] < 0 or point[0] >= 1000 or point[1] < 0 or point[1] >= 1000:
                fails.append(True)
                i += 1
                continue
            level = self.world.calc_level(point)
            penalty = self.world.has_level_penalty(level, base_level)
            diff_values.append(abs(level - base_level))
            values.append(level)
            penalty_prediction = self.calculate_lvl_pred(i)
            i += 1
            if penalty != penalty_prediction:
                fail = True
                fails.append(True)
            else:
                fails.append(False)
        if fail:
            print(f"WRONG PREDICTION! {fails}")
