# Maze generator -- Randomized Prim Algorithm

## Imports
import random
from colorama import init
from colorama import Fore, Back, Style
import cv2
import numpy as np
from datetime import datetime

wall = 'w'
cell = 'c'
unvisited = 'u'
 
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

def convertToObs(value):
    if value == 'w':
        return 1
    else:
        return 0

class Maze():

    def __init__(self, width, height):
        super()
        #Initialize colorama
        init()
        self.maze = []
        self.entrance = (0,0)
        self.exit = (0,0)
        self.pos = (0,0)
        self.height = height
        self.width = width

    def getObs(self):
        obs = np.zeros((self.height * self.width + 6,),dtype='uint8')
        #obs = np.zeros((6,),dtype='uint8')
        obs[0] = self.pos[0]
        obs[1] = self.pos[1]
        obs[2] = self.obsInPos((self.pos[0], self.pos[1] + 1))
        obs[3] = self.obsInPos((self.pos[0], self.pos[1] - 1))
        obs[4] = self.obsInPos((self.pos[0] + 1, self.pos[1]))
        obs[5] = self.obsInPos((self.pos[0] - 1, self.pos[1]))
        k = 2
        for i in range(0, self.height):
            for j in range(0, self.width):
                if (self.maze[i][j] == 'w'):
                    obs[k] = 1
                k += 1
        return obs

    ## Functions
    def render(self):
        img = np.zeros((50,50,3),dtype='uint8')
        for i in range(0, self.height):
            for j in range(0, self.width):
                if (self.maze[i][j] == 'w'):
                    img[i,j] = (0,0,255)
                else:
                    img[i,j] = (0,0,0)
                
        img[self.pos[0], self.pos[1]] = (255,255,255)
        cv2.imshow('a',img)
        cv2.waitKey(1)

    def move(self, direction):
        if direction == UP:
            newPos = (self.pos[0], self.pos[1] - 1)
        elif direction == DOWN:
            newPos = (self.pos[0], self.pos[1] + 1)
        elif direction == LEFT:
            newPos = (self.pos[0] - 1, self.pos[1])
        elif direction == RIGHT:
            newPos = (self.pos[0] + 1, self.pos[1])
        if self.canMove(newPos):
            self.pos = newPos

    def canMove(self, newPos):
        return newPos[0] >= 0 and newPos[0] < self.width and newPos[1] >= 0 and newPos[1] < self.height and self.maze[newPos[0]][newPos[1]] != 'w' 

    def obsInPos(self, newPos):
        return 1 if self.canMove(newPos) else 0

    # Find number of surrounding cells
    def surroundingCells(self,rand_wall):
        s_cells = 0
        if (self.maze[rand_wall[0]-1][rand_wall[1]] == 'c'):
            s_cells += 1
        if (self.maze[rand_wall[0]+1][rand_wall[1]] == 'c'):
            s_cells += 1
        if (self.maze[rand_wall[0]][rand_wall[1]-1] == 'c'):
            s_cells +=1
        if (self.maze[rand_wall[0]][rand_wall[1]+1] == 'c'):
            s_cells += 1

        return s_cells


    def generate(self):
        self.maze = []
        #random.seed(datetime.now().timestamp())
        # Denote all cells as unvisited
        for i in range(0, self.height):
            line = []
            for j in range(0, self.width):
                line.append(unvisited)
            self.maze.append(line)

        # Randomize starting point and set it a cell
        starting_height = int(random.random()*self.height)
        starting_width = int(random.random()*self.width)
        if (starting_height == 0):
            starting_height += 1
        if (starting_height == self.height-1):
            starting_height -= 1
        if (starting_width == 0):
            starting_width += 1
        if (starting_width == self.width-1):
            starting_width -= 1

        # Mark it as cell and add surrounding walls to the list
        self.maze[starting_height][starting_width] = cell
        walls = []
        walls.append([starting_height - 1, starting_width])
        walls.append([starting_height, starting_width - 1])
        walls.append([starting_height, starting_width + 1])
        walls.append([starting_height + 1, starting_width])

        # Denote walls in self.maze
        self.maze[starting_height-1][starting_width] = 'w'
        self.maze[starting_height][starting_width - 1] = 'w'
        self.maze[starting_height][starting_width + 1] = 'w'
        self.maze[starting_height + 1][starting_width] = 'w'

        while (walls):
            # Pick a random wall
            rand_wall = walls[int(random.random()*len(walls))-1]

            # Check if it is a left wall
            if (rand_wall[1] != 0):
                if (self.maze[rand_wall[0]][rand_wall[1]-1] == 'u' and self.maze[rand_wall[0]][rand_wall[1]+1] == 'c'):
                    # Find the number of surrounding cells
                    s_cells = self.surroundingCells(rand_wall)

                    if (s_cells < 2):
                        # Denote the new path
                        self.maze[rand_wall[0]][rand_wall[1]] = 'c'

                        # Mark the new walls
                        # Upper cell
                        if (rand_wall[0] != 0):
                            if (self.maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]-1, rand_wall[1]])

                        # Bottom cell
                        if (rand_wall[0] != self.height-1):
                            if (self.maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]+1, rand_wall[1]])

                        # Leftmost cell
                        if (rand_wall[1] != 0):    
                            if (self.maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                            if ([rand_wall[0], rand_wall[1]-1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]-1])

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)
                    continue

            # Check if it is an upper wall
            if (rand_wall[0] != 0):
                if (self.maze[rand_wall[0]-1][rand_wall[1]] == 'u' and self.maze[rand_wall[0]+1][rand_wall[1]] == 'c'):
                    s_cells = self.surroundingCells(rand_wall)
                    if (s_cells < 2):
                        # Denote the new path
                        self.maze[rand_wall[0]][rand_wall[1]] = 'c'

                        # Mark the new walls
                        # Upper cell
                        if (rand_wall[0] != 0):
                            if (self.maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]-1, rand_wall[1]])

                        # Leftmost cell
                        if (rand_wall[1] != 0):
                            if (self.maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                            if ([rand_wall[0], rand_wall[1]-1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]-1])

                        # Rightmost cell
                        if (rand_wall[1] != self.width-1):
                            if (self.maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                            if ([rand_wall[0], rand_wall[1]+1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]+1])

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)
                    continue

            # Check the bottom wall
            if (rand_wall[0] != self.height-1):
                if (self.maze[rand_wall[0]+1][rand_wall[1]] == 'u' and self.maze[rand_wall[0]-1][rand_wall[1]] == 'c'):
                    s_cells = self.surroundingCells(rand_wall)
                    if (s_cells < 2):
                        # Denote the new path
                        self.maze[rand_wall[0]][rand_wall[1]] = 'c'

                        # Mark the new walls
                        if (rand_wall[0] != self.height-1):
                            if (self.maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]+1, rand_wall[1]])
                        if (rand_wall[1] != 0):
                            if (self.maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                            if ([rand_wall[0], rand_wall[1]-1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]-1])
                        if (rand_wall[1] != self.width-1):
                            if (self.maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                            if ([rand_wall[0], rand_wall[1]+1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]+1])

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)
                    continue

            # Check the right wall
            if (rand_wall[1] != self.width-1):
                if (self.maze[rand_wall[0]][rand_wall[1]+1] == 'u' and self.maze[rand_wall[0]][rand_wall[1]-1] == 'c'):

                    s_cells = self.surroundingCells(rand_wall)
                    if (s_cells < 2):
                        # Denote the new path
                        self.maze[rand_wall[0]][rand_wall[1]] = 'c'

                        # Mark the new walls
                        if (rand_wall[1] != self.width-1):
                            if (self.maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                            if ([rand_wall[0], rand_wall[1]+1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]+1])
                        if (rand_wall[0] != self.height-1):
                            if (self.maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]+1, rand_wall[1]])
                        if (rand_wall[0] != 0):    
                            if (self.maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]-1, rand_wall[1]])

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)

                    continue

            # Delete the wall from the list anyway
            for wall in walls:
                if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                    walls.remove(wall)
            


        # Mark the remaining unvisited cells as walls
        for i in range(0, self.height):
            for j in range(0, self.width):
                if (self.maze[i][j] == 'u'):
                    self.maze[i][j] = 'w'

        # Set entrance and exit
        for i in range(0, self.width):
            if (self.maze[1][i] == 'c'):
                self.maze[0][i] = 'c'
                self.pos = self.entrance = (0, i)
                break

        for i in range(self.width-1, 0, -1):
            if (self.maze[self.height-2][i] == 'c'):
                self.maze[self.height-1][i] = 'c'
                self.exit = (self.height - 1, i)
                break
