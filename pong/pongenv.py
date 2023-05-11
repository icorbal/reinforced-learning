import pygame, sys, random
from pygame.locals import *
from fonts import *
import numpy as np
import gym
from gym import spaces


BLACK       = (0, 0, 0)
WHITE       = (255, 255, 255)
RED         = (205, 92, 92)
WIDTH       = 700
HEIGHT      = 500
BALL_RADIUS = 8
PAD_WIDTH   = 8
PAD_HEIGHT  = 70
DIFF         = 2         # positive integer representing difficulty level
TOPSCORE    = 10    

class Ball():
    def __init__(self, radius, color):
        self.radius = radius
        self.color  = color
    def reset(self, direction):
        self.pos = [int(WIDTH/2),int(HEIGHT/2)]
        if direction == "Right":
            self.vel = [random.randint(2,4),-random.randint(1,3)]
        elif direction == "Left":
            self.vel = [-random.randint(2,4),-random.randint(1,3)]
    def draw(self, window_obj):
        pygame.draw.circle(window_obj, self.color, self.pos, self.radius)
    def update(self):
        if self.pos[1] <= BALL_RADIUS or self.pos[1] >= HEIGHT-BALL_RADIUS:            # If the ball hits walls
            self.vel[1] = -self.vel[1]
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]    

class Paddle():
    def __init__(self, pos, vel, width, height, color, score):
        self.pos    = pos
        self.vel    = vel
        self.width  = width
        self.height = height
        self.color  = color
        self.score  = score
    def draw(self, window_obj):
        pygame.draw.line(window_obj, self.color, (self.pos[0], self.pos[1] - self.height/2), (self.pos[0], self.pos[1] + self.height/2), self.width)
    def update(self):
        if self.pos[1]+self.vel > int(self.height/2) and self.pos[1]+self.vel < int(HEIGHT-self.height/2):
            self.pos[1]=self.pos[1]+self.vel 

def playerPaddleHeight(self):
    if not self.rendering:
        return HEIGHT
    else:
        return PAD_HEIGHT

def check_collision(self, ball, paddle1, paddle2):
    if ball.pos[0] <= paddle1.width*2 + ball.radius or ball.pos[0] >= WIDTH - paddle1.width*2 - ball.radius:
        if (ball.pos[1] in range(paddle1.pos[1] - int(paddle1.height/2), paddle1.pos[1] + int(paddle1.height/2)) and ball.pos[0] > int(WIDTH/2)) or (ball.pos[1] in range(paddle2.pos[1] - int(paddle2.height/2), paddle2.pos[1] + int(paddle2.height/2)) and ball.pos[0] < int(WIDTH/2)):
            if ball.vel[0]<0:
                ball.vel[0] =- (ball.vel[0]-1)
                self.reward = 5
                self.hit_estimation = estimate_ball_hit(ball, paddle2)
            else:
                ball.vel[0] =- (ball.vel[0]+1)

        else:
            if ball.pos[0]>int(WIDTH/2):        #check which player gets a point depending on which side the ball is in
                paddle1.score +=1
                ball.reset("Left")
                self.reward = -10
            else:
                paddle2.score +=1
                ball.reset("Right")
            self.hit_estimation = estimate_ball_hit(ball, paddle2)

def estimate_ball_hit(ball, paddle):
    proj = Ball(BALL_RADIUS, RED)
    proj.pos = [ball.pos[0], ball.pos[1]]
    proj.vel = [ball.vel[0], ball.vel[1]]

    while proj.vel[0] > 0 or proj.pos[0] > paddle.width*2 + proj.radius:
        proj.update()
        if proj.pos[0] >= WIDTH - paddle.width*2 - proj.radius:
            proj.vel[0] = -proj.vel[0] - 1
    return proj.pos[1]
    

def win_msg(window_obj, player):
    window_obj.fill(BLACK)
    try:
        font             = pygame.font.Font("fonts/Megadeth.ttf", 70)
    except:
        font             = pygame.font.Font(None, 70)
    msg             = font.render(player + " Wins", True, WHITE)
    msgRect         = msg.get_rect()
    msgRect.centerx = int(WIDTH/2)
    msgRect.centery = int(HEIGHT/2)
    window_obj.blit(msg, msgRect)
    pygame.display.update()
    while True:    
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                return

def draw_scores(window_obj, score1, score2):
    try:
        font          = pygame.font.Font("fonts/impact.ttf", 40)
    except:
        font          = pygame.font.Font(None, 40)
    msg1          = font.render(str(score1), True, WHITE)
    msg2          = font.render(str(score2), True, WHITE)    
    msg1Rect      = msg1.get_rect()
    msg2Rect      = msg2.get_rect()
    msg1Rect.left = int(WIDTH/4)
    msg2Rect.left = int(WIDTH/4 * 3)
    msg1Rect.top  = int(HEIGHT/4)
    msg2Rect.top  = int(HEIGHT/4)
    window_obj.blit(msg1, msg1Rect)
    window_obj.blit(msg2, msg2Rect)

def run_game(self, ball, paddle1, paddle2):
    #pygame.draw.line(window, WHITE, (int(WIDTH/2), 0), (int(WIDTH/2), HEIGHT), 2)    #Draw central line and borders
    #pygame.draw.line(window, WHITE, (PAD_WIDTH, 0), (PAD_WIDTH, HEIGHT), 2)
    #pygame.draw.line(window, WHITE, (WIDTH - PAD_WIDTH, 0), (WIDTH - PAD_WIDTH, HEIGHT), 2)
    ball.update()
    paddle1.update()
    paddle2.update()
    check_collision(self, ball, paddle1, paddle2)
    
def createObservation(self):
    return [self.hit_estimation, self.paddle2.pos[1]]

class PongEnv(gym.Env):

    def __init__(self):
        super(PongEnv, self).__init__()
        self.rendering = False
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-700, high=700,
                                            shape=(2,), dtype=np.float32)

    def step(self, action):
        ball = self.ball
        paddle1 = self.paddle1
        paddle2 = self.paddle2
        self.reward = 0
        run_game(self, ball, paddle1, paddle2)
        if paddle1.score >= TOPSCORE:
            self.done = True
        elif paddle2.score >= TOPSCORE:
            self.done = True
        if action == 1:
            paddle2.vel = DIFF
        elif action == 2:
            paddle2.vel = -DIFF
        self.observation = createObservation(self)
        return self.observation, self.reward, self.done, {}

    def reset(self):
        ball = Ball(BALL_RADIUS, RED)
        ball.reset("Left")
        paddle1 = Paddle([int(WIDTH-PAD_WIDTH*2),int(HEIGHT/2)], 0, PAD_WIDTH, playerPaddleHeight(self), WHITE, 0)
        paddle2 = Paddle([int(PAD_WIDTH*2),int(HEIGHT/2)], 0, PAD_WIDTH, PAD_HEIGHT, WHITE, 0)
        self.hit_estimation = estimate_ball_hit(ball, paddle2)
        self.ball = ball
        self.paddle1 = paddle1
        self.paddle2 = paddle2
        self.observation = createObservation(self)
        self.done = False
        return self.observation
       
    def initRendering(self):
        pygame.init()
        self.fpsClock = pygame.time.Clock()
        self.window_obj = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        pygame.display.set_caption("Pong")
        self.rendering = True


    def render(self):
        self.window_obj.fill(BLACK)                            #clear screen before drawing again
        self.ball.draw(self.window_obj)
        self.paddle1.draw(self.window_obj)
        self.paddle2.draw(self.window_obj)
        draw_scores(self.window_obj, self.paddle1.score, self.paddle2.score)
        pygame.display.update()
        if self.paddle1.score >= TOPSCORE:
            win_msg(self.window_obj, 'Player 1')
        elif self.paddle2.score >= TOPSCORE:
            win_msg(self.window_obj, 'Player 2')
        for event in pygame.event.get():            #event handler
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:    
                if event.key == K_ESCAPE:
                    self.done = True
                    return
                elif event.key == K_UP:
                        self.paddle1.vel = -3
                elif event.key == K_DOWN:
                        self.paddle1.vel = 3                
            elif event.type == KEYUP:
                if event.key == K_UP:
                        self.paddle1.vel = 0
                elif event.key == K_DOWN:
                        self.paddle1.vel = 0
        self.fpsClock.tick(60)                        #run at 60 fps
