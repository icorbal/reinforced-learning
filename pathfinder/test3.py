import numpy as np


def get_level_info(param, base_level):
    return param


def get_levels(matrix, exploration_radius, x, y, base_level):
    height, width = matrix.shape
    values = []
    for radius in range(exploration_radius + 1):
        for i in range(x - radius, x + radius + 1):
            if 0 <= i < height and 0 <= y - radius < width:
                if not (i == x and y - radius == y):
                    values.append(get_level_info(matrix[i, y - radius], base_level))  # Top side
            else:
                values.append(1)

        for j in range(y - radius + 1, y + radius + 1):
            if 0 <= x + radius < height and 0 <= j < width:
                if not (x + radius == x and j == y):
                    values.append(get_level_info(matrix[x + radius, j], base_level))  # Right side
            else:
                values.append(1)

        for i in range(x + radius - 1, x - radius - 1, -1):
            if 0 <= i < height and 0 <= y + radius < width:
                if not (i == x and y + radius == y):
                    values.append(get_level_info(matrix[i, y + radius], base_level))  # Bottom side
            else:
                values.append(1)

        for j in range(y + radius - 1, y - radius, -1):
            if 0 <= x - radius < height and 0 <= j < width:
                if not (x - radius == x and j == y):
                    values.append(get_level_info(matrix[x - radius, j], base_level))  # Left side
            else:
                values.append(1)
    return values
# Create a sample matrix
matrix = np.array([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20]])

# Starting point
start_x, start_y = 2, 2

# Iterate through concentric squares
for value in get_levels(matrix, 1, start_x, start_y, 0):
    print(value)
