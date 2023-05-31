from time import sleep
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
MAX_ITER = 10000


def action_from_key(key):
    if 48 <= key <= 56:
        return key - 48
    else:
        return None


def matrix_diff(matrix1, matrix2):
    difference = np.abs(matrix1 - matrix2)
    return np.mean(difference ** 2)


def dist_to_goal(point):
    return 4 - np.sqrt((point.x - 1) ** 2 + (point.y - 3) ** 2)


def get_num_id(x):
    return x.num_id if x else 0


class KhunPanEnv(gym.Env):

    def __init__(self):
        super(KhunPanEnv, self).__init__()
        self.num_moves = 0
        self.count_original = 0
        self.count_repeated = 0
        self.count_invalid = 0
        self.max_height = 0
        self.num_success = 0
        self.reward = None
        self.observation = np.zeros((5 * 4,), dtype='uint8')
        self.done = False
        self.m_hashes = set()
        self.board = Board()
        self.img = None
        self.init_obs = None
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=4,
                                            shape=(5 * 4,), dtype=np.uint8)

    def step(self, action):
        piece_moved = self.process_action(action)
        observations = self.board.get_observations()
        self.reward = 0
        if piece_moved is not None:
            m_hash = hash(observations.tostring())
            if m_hash not in self.m_hashes:
                if self.init_obs is None:
                    self.init_obs = observations
                else:
                    self.reward = matrix_diff(observations, self.init_obs) + dist_to_goal(self.board.square4s[0])
                self.m_hashes.add(m_hash)
                self.count_original += 1
            else:
                self.count_repeated += 1
        else:
            self.count_invalid += 1
        if self.board.square4s[0].y > self.max_height:
            self.max_height = self.board.square4s[0].y
        if self.board.square4s[0].x == 1 and self.board.square4s[0].y == 3:
            self.done = True
            self.reward = 10000
            self.num_success += 1
        if self.num_moves > MAX_ITER:
            self.done = True
        self.num_moves += 1
        return observations, self.reward, self.done, False, {}

    def reset(self, **kwargs):
        self.img = np.zeros((500, 400, 3), dtype='uint8')
        self.done = False
        info = {
            "unique_moves": self.count_original,
            "repeated_moves": self.count_repeated,
            "invalid_moves": self.count_invalid,
            "max_height": self.max_height,
            "num_success": self.num_success
        }
        self.num_moves = 0
        self.count_original = 0
        self.count_repeated = 0
        self.count_invalid = 0
        self.max_height = 0
        self.num_success = 0
        self.init_obs = None
        self.m_hashes = set()
        self.board = Board()
        return self.board.get_observations(), info

    def render(self):
        cv2.imshow('a', self.img)
        cv2.waitKey(1)
        sleep(0.1)
        # self.detect_key()
        self.img = np.zeros((500, 400, 3), dtype='uint8')
        self.board.render(self.img)

    def detect_key(self):
        key = cv2.waitKey(0)
        action = action_from_key(key)
        if action is not None:
            self.process_action(action)

    def process_action(self, action):
        possible_actions = self.get_possible_actions()
        num_actions = len(possible_actions)
        if num_actions > 7:
            print("possible_actions length greater than 7")
        piece_action = possible_actions[num_actions - 1] if action >= num_actions else possible_actions[action]
        if piece_action[0].try_move(piece_action[1]):
            return piece_action[0]
        else:
            return None

    def try_move_pos(self, x, y, action):
        if action == LEFT:
            if x < 3:
                piece = self.board.matrix[x + 1][y]
                if piece is not None:
                    if piece.try_move(LEFT):
                        return piece
                    else:
                        return None
        elif action == RIGHT:
            if x > 0:
                piece = self.board.matrix[x - 1][y]
                if piece is not None:
                    if piece.try_move(RIGHT):
                        return piece
                    else:
                        return None
        elif action == UP:
            if y < 4:
                piece = self.board.matrix[x][y + 1]
                if piece is not None:
                    if piece.try_move(UP):
                        return piece
                    else:
                        return None
        elif action == DOWN:
            if y > 0:
                piece = self.board.matrix[x][y - 1]
                if piece is not None:
                    if piece.try_move(DOWN):
                        return piece
                    else:
                        return None
        else:
            return None

    def check_move_pos(self, x, y, action):
        if action == LEFT:
            if x < 3:
                piece = self.board.matrix[x + 1][y]
                if piece is not None:
                    if piece.check_move(LEFT):
                        return piece
                    else:
                        return None
        elif action == RIGHT:
            if x > 0:
                piece = self.board.matrix[x - 1][y]
                if piece is not None:
                    if piece.check_move(RIGHT):
                        return piece
                    else:
                        return None
        elif action == UP:
            if y < 4:
                piece = self.board.matrix[x][y + 1]
                if piece is not None:
                    if piece.check_move(UP):
                        return piece
                    else:
                        return None
        elif action == DOWN:
            if y > 0:
                piece = self.board.matrix[x][y - 1]
                if piece is not None:
                    if piece.check_move(DOWN):
                        return piece
                    else:
                        return None
        else:
            return None

    def get_possible_actions(self):
        possible_actions = []
        x, y = self.board.find_empty_space(True)
        for action in range(4):
            piece = self.check_move_pos(x, y, action)
            if piece is not None:
                possible_actions.append((piece, action))
        x, y = self.board.find_empty_space(False)
        for action in range(4):
            piece = self.check_move_pos(x, y, action)
            if piece is not None:
                possible_actions.append((piece, action))
        return possible_actions


class Board:
    def __init__(self):
        self.matrix = np.full((4, 5), None)
        self.square4s = [
            Square4(matrix=self.matrix, x=1, y=0)
        ]
        self.rectanglev2s = [
            RectangleV2(matrix=self.matrix, x=0, y=0), RectangleV2(matrix=self.matrix, x=0, y=2),
            RectangleV2(matrix=self.matrix, x=3, y=0), RectangleV2(matrix=self.matrix, x=3, y=2)
        ]
        self.rectangleh2s = [
            RectangleH2(matrix=self.matrix, x=1, y=2)
        ]
        self.square1s = [
            Square1(matrix=self.matrix, x=1, y=3), Square1(matrix=self.matrix, x=1, y=4),
            Square1(matrix=self.matrix, x=2, y=3), Square1(matrix=self.matrix, x=2, y=4),
        ]

    def render(self, img):
        for square1 in self.square1s:
            square1.render(img)
        for square4 in self.square4s:
            square4.render(img)
        for rectanglev2 in self.rectanglev2s:
            rectanglev2.render(img)
        for rectangleh2 in self.rectangleh2s:
            rectangleh2.render(img)

    def find_empty_space(self, first=True):
        found_first = False
        for x in range(4):
            for y in range(5):
                if self.matrix[x][y] is None:
                    if first or found_first is True:
                        return x, y
                    else:
                        found_first = True
        print("Could not find empty space (first={})".format(first))
        print(self.matrix)
        return None, None

    def get_observations(self):
        observations = np.zeros((4 * 5), dtype='uint8')
        i = 0
        for x in range(4):
            for y in range(5):
                if self.matrix[x][y] is not None:
                    observations[i] = self.matrix[x][y].num_id
                i = i + 1
        return observations


class Piece:
    def __init__(self, num_id, matrix, x, y, w, h, color):
        self.num_id = num_id
        self.matrix = matrix
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color

    def render(self, img):
        cv2.rectangle(img, (self.x * 100 + 3, self.y * 100 + 3),
                      (self.x * 100 + self.w * 100 - 3, self.y * 100 + self.h * 100 - 3), self.color, 3)

    def try_move(self, movement):
        if self.check_move(movement):
            self.do_move(movement)
            return True
        else:
            return False

    def do_move(self, movement):
        pass

    def check_move(self, movement):
        if movement == LEFT:
            return self.x > 0 and (
                    (self.h == 1 and self.matrix[self.x - 1][self.y] is None)
                    or (self.h == 2 and self.matrix[self.x - 1][self.y] is None
                        and self.matrix[self.x - 1][self.y + 1] is None)
            )
        elif movement == RIGHT:
            return self.x + self.w < 4 and (
                    (self.h == 1 and self.matrix[self.x + self.w][self.y] is None)
                    or (self.h == 2 and self.matrix[self.x + self.w][self.y] is None
                        and self.matrix[self.x + self.w][self.y + 1] is None)
            )
        elif movement == UP:
            return self.y > 0 and (
                    (self.w == 1 and self.matrix[self.x][self.y - 1] is None)
                    or (self.w == 2 and self.matrix[self.x][self.y - 1] is None
                        and self.matrix[self.x + 1][self.y - 1] is None)
            )
        elif movement == DOWN:
            return self.y + self.h < 5 and (
                    (self.w == 1 and self.matrix[self.x][self.y + self.h] is None)
                    or (self.w == 2 and self.matrix[self.x][self.y + self.h] is None
                        and self.matrix[self.x + 1][self.y + self.h] is None)
            )


class Square1(Piece):
    def __init__(self, matrix, num_id=1, x=0, y=0):
        super().__init__(num_id=num_id, matrix=matrix, x=x, y=y, w=1, h=1, color=(255, 0, 0))
        self.matrix[self.x][self.y] = self

    def do_move(self, movement):
        if movement == LEFT:
            self.x -= 1
            self.matrix[self.x + 1][self.y] = None
        elif movement == RIGHT:
            self.x += 1
            self.matrix[self.x - 1][self.y] = None
        elif movement == UP:
            self.y -= 1
            self.matrix[self.x][self.y + 1] = None
        elif movement == DOWN:
            self.y += 1
            self.matrix[self.x][self.y - 1] = None
        self.matrix[self.x][self.y] = self


class Square4(Piece):
    def __init__(self, matrix, num_id=4, x=0, y=0):
        super().__init__(num_id=num_id, matrix=matrix, x=x, y=y, w=2, h=2, color=(255, 255, 255))
        self.matrix[self.x][self.y] = self
        self.matrix[self.x + 1][self.y] = self
        self.matrix[self.x][self.y + 1] = self
        self.matrix[self.x + 1][self.y + 1] = self

    def do_move(self, movement):
        if movement == LEFT:
            self.x -= 1
            self.matrix[self.x + 2][self.y] = None
            self.matrix[self.x + 2][self.y + 1] = None
        elif movement == RIGHT:
            self.x += 1
            self.matrix[self.x - 1][self.y] = None
            self.matrix[self.x - 1][self.y + 1] = None
        elif movement == UP:
            self.y -= 1
            self.matrix[self.x][self.y + 2] = None
            self.matrix[self.x + 1][self.y + 2] = None
        elif movement == DOWN:
            self.y += 1
            self.matrix[self.x][self.y - 1] = None
            self.matrix[self.x + 1][self.y - 1] = None
        self.matrix[self.x][self.y] = self
        self.matrix[self.x + 1][self.y] = self
        self.matrix[self.x][self.y + 1] = self
        self.matrix[self.x + 1][self.y + 1] = self


class RectangleV2(Piece):
    def __init__(self, matrix, num_id=3, x=0, y=0):
        super().__init__(num_id=num_id, matrix=matrix, x=x, y=y, w=1, h=2, color=(0, 0, 255))
        self.matrix[self.x][self.y] = self
        self.matrix[self.x][self.y + 1] = self

    def do_move(self, movement):
        if movement == LEFT:
            self.x -= 1
            self.matrix[self.x + 1][self.y] = None
            self.matrix[self.x + 1][self.y + 1] = None
        elif movement == RIGHT:
            self.x += 1
            self.matrix[self.x - 1][self.y] = None
            self.matrix[self.x - 1][self.y + 1] = None
        elif movement == UP:
            self.y -= 1
            self.matrix[self.x][self.y + 2] = None
        elif movement == DOWN:
            self.y += 1
            self.matrix[self.x][self.y - 1] = None
        self.matrix[self.x][self.y] = self
        self.matrix[self.x][self.y + 1] = self


class RectangleH2(Piece):
    def __init__(self, matrix, num_id=2, x=0, y=0):
        super().__init__(num_id=num_id, matrix=matrix, x=x, y=y, w=2, h=1, color=(0, 0, 255))
        self.matrix[self.x][self.y] = self
        self.matrix[self.x + 1][self.y] = self

    def do_move(self, movement):
        if movement == LEFT:
            self.x -= 1
            self.matrix[self.x + 2][self.y] = None
        elif movement == RIGHT:
            self.x += 1
            self.matrix[self.x - 1][self.y] = None
        elif movement == UP:
            self.y -= 1
            self.matrix[self.x][self.y + 1] = None
            self.matrix[self.x + 1][self.y + 1] = None
        elif movement == DOWN:
            self.y += 1
            self.matrix[self.x][self.y - 1] = None
            self.matrix[self.x + 1][self.y - 1] = None
        self.matrix[self.x][self.y] = self
        self.matrix[self.x + 1][self.y] = self
