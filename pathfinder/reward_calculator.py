import math


class RewardCalculator:
    def __init__(self, world, initial_dist, destination_point, max_level_diff):
        self.prev_dist_prop = None
        self.base_level = None
        self.prev_dist = None
        self.world = world
        self.initial_dist = initial_dist
        self.destination_point = destination_point
        self.max_level_diff = max_level_diff
        
    def update_current_point(self, current_point):
        self.prev_dist = math.dist(self.destination_point, current_point)
        self.prev_dist_prop = self.prev_dist / self.initial_dist
        self.base_level = self.world.calc_level(current_point)

    def calculate_reward(self, point):
        if self.has_level_penalty(point):
            return -5
        else:
            dist = math.dist(self.destination_point, point)
            dist_prop = dist / self.initial_dist
            dist_bonus = 1 - dist_prop
            if dist_bonus < 0:
                dist_bonus = 0
            reward = (self.prev_dist_prop - dist_prop) * 10 + dist_bonus
            if point == self.destination_point:
                reward = 10000
        return reward

    def has_level_penalty(self, point):
        level = self.world.calc_level(point)
        level_diff = abs(level - self.base_level)
        if level_diff > self.max_level_diff:
            return True
        else:
            return False
