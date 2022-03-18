# our environement here is adapted from: https://github.com/TheAILearner/Snake-Game-using-OpenCV-Python/blob/master/snake_game_using_opencv.ipynb
import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_INITIAL_LENGTH = 3
MAX_HITS = 0
AVOID_SNAKE_PIT = 1
def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
    score += 1
    return apple_position, score

def collision_with_boundaries(snake_head):
    if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0

def countFree(position, snake_position, explored, free):
    if len(free) >= len(snake_position) or position in explored or position in snake_position or collision_with_boundaries(position):
        return len(free)
    else:
        free.append(position)
        explored.append(position)
        countFree([position[0]-10, position[1]], snake_position, explored, free)
        countFree([position[0], position[1]-10], snake_position, explored, free)
        countFree([position[0]+10, position[1]], snake_position, explored, free)
        countFree([position[0], position[1]+10], snake_position, explored, free)
        return len(free)

def inSnakeTrap(position, snake_position):
    explored = []
    free = []
    if countFree(position, snake_position, explored, free) >= len(snake_position):
        return False
    else:
        return True


def predictCollision(snake_position, position):
    if (position in snake_position[1:] or 
            collision_with_boundaries(position) or 
            (AVOID_SNAKE_PIT and inSnakeTrap(position, snake_position))):
        return 0
    else:
        return 100

def createObservation(self):
    head_x = self.snake_head[0]
    head_y = self.snake_head[1]

    #snake_length = len(self.snake_position)
    apple_delta_x = self.apple_position[0] - head_x
    apple_delta_y = self.apple_position[1] - head_y

    left_collision = predictCollision(self.snake_position, [head_x - 10, head_y]) 
    top_collision = predictCollision(self.snake_position, [head_x, head_y - 10]) 
    right_collision = predictCollision(self.snake_position, [head_x + 10, head_y]) 
    bottom_collision = predictCollision(self.snake_position, [head_x, head_y + 10]) 
    observation = [apple_delta_y, apple_delta_x, left_collision, top_collision, right_collision, bottom_collision]
    return observation

def waitTime(timeInSeconds):
    t_end = time.time() + timeInSeconds
    k = -1
    while time.time() < t_end:
        if k == -1:
            k = cv2.waitKey(1)
        else:
            continue

def createInitialSnake():
    snake = []
    for i in range(SNAKE_INITIAL_LENGTH):
        snake.append([250, 250 - i * 10])
    return snake

class SnekEnv(gym.Env):

    def __init__(self):
        super(SnekEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(6,), dtype=np.float32)


    def step(self, action):
        button_direction = action
        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        # Increase Snake length on eating apple
        apple_reward = 0
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0,list(self.snake_head))
            apple_reward = 100
        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
        
        # On collision kill the snake and print the score
        hit = False
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            hit = True
            if self.num_hits >= MAX_HITS:
                self.done = True
                #print("Score: " + str(self.score))
            else:
                self.num_hits = self.num_hits + 1
            #print(str(format(self.observation[2], "04b")))
            #print(str(self.snake_position[0][0] - self.snake_position[1][0]) + ", " + str(self.snake_position[0][1] - self.snake_position[1][1]))
            #time.sleep(10)

        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        stage = (len(self.snake_position) - SNAKE_INITIAL_LENGTH)
        self.reward = (stage * 10 + 250 - euclidean_dist_to_apple) / 250 + apple_reward

        #print(self.reward)

        #self.reward = self.total_reward - self.prev_reward
        #self.prev_reward = self.total_reward

        if hit:
            #print(self.reward)
            self.reward = -10 * stage
            #print(self.observation)
            #print(euclidean_dist_to_apple)
            #print(str(self.snake_position[0][0] - self.snake_position[1][0]) + ", " + str(self.snake_position[0][1] - self.snake_position[1][1]))
        info = {}

        self.observation = createObservation(self)

        return self.observation, self.reward, self.done, info

    def reset(self):
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = createInitialSnake()
        self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250,250]
        self.num_hits = 0
        self.done = False

        self.observation = createObservation(self)
        return self.observation

    def render(self):
        cv2.imshow('a',self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500,500,3),dtype='uint8')
        
        # Display Apple
        cv2.rectangle(self.img,(self.apple_position[0],self.apple_position[1]),(self.apple_position[0]+10,self.apple_position[1]+10),(0,0,255),3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)
    
        # Takes step after fixed time
        waitTime(0.05)
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500,500,3),dtype='uint8')
            cv2.putText(self.img,'Your Score is {}'.format(self.score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('a',self.img)
