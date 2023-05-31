import noise
import numpy as np

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
UP_LEFT = 4
UP_RIGHT = 5
DOWN_LEFT = 6
DOWN_RIGHT = 7


def calculate_next_point(point, action, distance=1):
    if action == LEFT:
        return point[0] - distance, point[1]
    elif action == RIGHT:
        return point[0] + distance, point[1]
    elif action == UP:
        return point[0], point[1] + distance
    elif action == DOWN:
        return point[0], point[1] - distance
    elif action == UP_LEFT:
        return point[0] - distance, point[1] + distance
    elif action == UP_RIGHT:
        return point[0] + distance, point[1] + distance
    elif action == DOWN_LEFT:
        return point[0] - distance, point[1] - distance
    elif action == DOWN_RIGHT:
        return point[0] + distance, point[1] - distance


class World:
    def __init__(self, max_level_diff):
        shape = (1000, 1000)
        scale = 1500.0
        octaves = 2
        persistence = 5.5
        lacunarity = 8.0
        self.max_level_diff = max_level_diff
        self.z = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.z[i][j] = noise.pnoise2(i / scale,
                                             j / scale,
                                             octaves=octaves,
                                             persistence=persistence,
                                             lacunarity=lacunarity,
                                             repeatx=1024,
                                             repeaty=1024,
                                             base=82) / 4
        self.lin_x = np.linspace(0, 1, shape[0], endpoint=False)
        self.lin_y = np.linspace(0, 1, shape[1], endpoint=False)
        self.x, self.y = np.meshgrid(self.lin_x, self.lin_y)

    def calc_level(self, point):
        return self.z[point[1]][point[0]]

    def get_levels(self, exploration_radius, x, y, reward_calculator):
        matrix = self.z
        height, width = matrix.shape
        penalties = []
        for radius in range(exploration_radius + 1):
            for i in range(x - radius, x + radius + 1):
                if 0 <= i < height and 0 <= y - radius < width:
                    if not (i == x and y - radius == y):
                        penalties.append(reward_calculator.calculate_reward((i, y - radius)))
                else:
                    penalties.append(0)

            for j in range(y - radius + 1, y + radius + 1):
                if 0 <= x + radius < height and 0 <= j < width:
                    if not (x + radius == x and j == y):
                        penalties.append(reward_calculator.calculate_reward((x + radius, j)))
                else:
                    penalties.append(0)

            for i in range(x + radius - 1, x - radius - 1, -1):
                if 0 <= i < height and 0 <= y + radius < width:
                    if not (i == x and y + radius == y):
                        penalties.append(reward_calculator.calculate_reward((i, y + radius)))
                else:
                    penalties.append(0)

            for j in range(y + radius - 1, y - radius, -1):
                if 0 <= x - radius < height and 0 <= j < width:
                    if not (x - radius == x and j == y):
                        penalties.append(reward_calculator.calculate_reward((x - radius, j)))
                else:
                    penalties.append(0)
        return penalties
