import noise
import numpy as np
from numpy import array, uint8

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
UP_LEFT = 4
UP_RIGHT = 5
DOWN_LEFT = 6
DOWN_RIGHT = 7

WORLD_SIZE = 1000


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


def convert_255(element):
    return int(128 + element * 128)


class World:
    def __init__(self, max_level_diff, obs_dim):
        shape = (WORLD_SIZE, WORLD_SIZE)
        scale = 1500.0
        octaves = 2
        persistence = 5.5
        lacunarity = 8.0
        self.walked_points = []
        self.max_level_diff = max_level_diff
        self.obs_dim = obs_dim
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
        self.mean_img = self.calc_mean_image()

    def calc_level(self, point):
        return self.z[point[0]][point[1]]

    def reset(self, destination_point):
        self.walked_points = []

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

    def calc_snapshot(self, dest_point):
        layer = np.vectorize(convert_255)(self.z.copy())
        layer[dest_point[0]][dest_point[1]] = 255
        return layer

    def calc_mean_image(self):
        default_size = len(self.z)
        diff_size = default_size // self.obs_dim
        reduced_array = self.z.reshape((self.obs_dim, diff_size, self.obs_dim, diff_size))
        mean_array = reduced_array.mean(axis=(1, 3))
        mean_array_255 = np.vectorize(convert_255)(mean_array)
        return mean_array_255.astype(uint8)

    def mean_obs(self, current_point):
        default_size = len(self.mean_img)
        size_prop = len(self.z) // default_size
        snapshot = np.zeros((self.obs_dim, self.obs_dim))
        x_pos = current_point[0] // size_prop
        y_pos = current_point[1] // size_prop
        if x_pos >= self.obs_dim // 2:
            x1 = 0
            if default_size - x_pos <= self.obs_dim // 2:
                x2 = self.obs_dim // 2 + default_size - x_pos
                x3 = x_pos - self.obs_dim // 2
                x4 = default_size
            else:
                x2 = self.obs_dim
                x3 = x_pos - self.obs_dim // 2
                x4 = self.obs_dim // 2 + x_pos
        else:
            if x_pos <= self.obs_dim // 2:
                x1 = self.obs_dim // 2 - x_pos
                x2 = self.obs_dim
                x3 = 0
                x4 = self.obs_dim // 2 + x_pos
            else:
                x1 = 0
                x2 = self.obs_dim
                x3 = x_pos - self.obs_dim // 2
                x4 = x3 + self.obs_dim
        if y_pos >= self.obs_dim // 2:
            y1 = 0
            if default_size - y_pos <= self.obs_dim // 2:
                y2 = self.obs_dim // 2 + default_size - y_pos
                y3 = y_pos - self.obs_dim // 2
                y4 = default_size
            else:
                y2 = self.obs_dim
                y3 = y_pos - self.obs_dim // 2
                y4 = self.obs_dim // 2 + y_pos
        else:
            if y_pos <= self.obs_dim // 2:
                y1 = self.obs_dim // 2 - y_pos
                y2 = self.obs_dim
                y3 = 0
                y4 = self.obs_dim // 2 + y_pos
            else:
                y1 = 0
                y2 = self.obs_dim
                y3 = y_pos - self.obs_dim // 2
                y4 = y3 + self.obs_dim
        snapshot[x1:x2, y1:y2] = self.mean_img[x3:x4, y3:y4]
        return snapshot.astype(np.uint8)

    def take_snapshot(self, current_point, destination_point):
        snapshot = np.zeros((self.obs_dim, self.obs_dim)).astype(np.uint8)
        x = destination_point[0] - current_point[0] + self.obs_dim // 2
        y = destination_point[1] - current_point[1] + self.obs_dim // 2
        if x < 0:
            x = 0
        elif x >= self.obs_dim:
            x = self.obs_dim - 1
        if y < 0:
            y = 0
        elif y >= self.obs_dim:
            y = self.obs_dim - 1
        self.draw_point(snapshot, x, y, 20)
        return array([snapshot])#, self.mean_obs(current_point)])

    @staticmethod
    def draw_point(matrix, x, y, point_radius):
        height, width = matrix.shape
        matrix[y][x] = 255
        for radius in range(point_radius + 1):
            for i in range(y - radius, y + radius + 1):
                if 0 <= i < height and 0 <= x - radius < width:
                    if not (i == y and x - radius == x):
                        matrix[i, x - radius] = 255

            for j in range(x - radius + 1, x + radius + 1):
                if 0 <= y + radius < height and 0 <= j < width:
                    if not (y + radius == y and j == x):
                        matrix[y + radius, j] = 255

            for i in range(y + radius - 1, y - radius - 1, -1):
                if 0 <= i < height and 0 <= x + radius < width:
                    if not (i == y and x + radius == x):
                        matrix[i, x + radius] = 255

            for j in range(x + radius - 1, x - radius, -1):
                if 0 <= y - radius < height and 0 <= j < width:
                    if not (y - radius == y and j == x):
                        matrix[y - radius, j] = 255
