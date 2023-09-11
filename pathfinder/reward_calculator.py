import math


class RewardCalculator:
    def __init__(self, world, initial_dist, destination_point, max_level_diff):
        self.prev_dist_prop = None
        self.base_level = None
        self.prev_dist = None
        self.min_dist = initial_dist
        self.world = world
        self.initial_dist = initial_dist
        self.destination_point = destination_point
        self.max_level_diff = max_level_diff
        
    def update_current_point(self, current_point):
        self.prev_dist = math.dist(self.destination_point, current_point)
        self.prev_dist_prop = self.prev_dist / self.initial_dist
        self.base_level = self.world.calc_level(current_point)

    def calculate_reward(self, point):
        #if point in self.world.walked_points:
        #   return -5
        #level = self.world.calc_level(point)
        level_penalty = 0 #- abs(level - self.base_level) * 4000
        if point[0] < 0 or point[0] >= 1000 or point[1] < 0 or point[1] >= 1000:
            return -5
        dist = math.dist(self.destination_point, point)
        dist_prop = dist / self.initial_dist
        dist_bonus = 1 if dist < self.min_dist else 0
        if dist_prop > 1:
            dist_prop = 0
        reward = dist_prop * dist_prop * dist_bonus
        #reward = (self.prev_dist_prop - dist_prop) * 10 + dist_bonus * 2 + level_penalty
        self.min_dist = min(self.min_dist, dist)
        return reward

    def has_level_penalty(self, point):
        level = self.world.calc_level(point)
        level_diff = abs(level - self.base_level)
        if level_diff > self.max_level_diff:
            return True
        else:
            return False
