import chess
import chess.svg
import chess.engine
import pygame
import cairosvg
import numpy as np
from pygame.locals import *
import gym
from gym import spaces
from PIL import Image
from io import BytesIO
from stockfish import Stockfish
from collections import deque

WINDOW_SIZE= 500
MAX_INCORRECT_MOVES=500
MAX_LOG=10
SOLO_PLAY=False
MAX_MOVES=30
def getBoardStatus(self):
    if self.board.is_checkmate():
        return 2
    elif self.board.is_check():
        return 1
    elif self.board.is_stalemate():
        return 3
    elif self.board.is_seventyfive_moves():
        return 4
    elif self.board.is_fivefold_repetition():
        return 5
    elif self.board.is_insufficient_material():
        return 6
    else:
        return 0

def calculateReward(self):
    self.reward = 0
    #if self.lastMove is None and self.previousMove is not None:
    #    self.reward = -1
    if self.lastMove is None:
        self.reward = -10 / (self.numMoves + 1)
    elif self.previousStatus == 2:
        self.reward = 10
    elif self.previousStatus == 1:
        self.reward = 3
    elif self.previousStatus > 2:
        self.reward = 5
    elif self.lastStatus == 2:
        self.reward = -50
    elif self.lastStatus == 1:
        self.reward = -5
    elif self.previousStatus > 2 or self.lastStatus > 2:
        self.reward = 5
    elif self.currentBalance != 0:
        self.reward = self.currentBalance
    else:
        self.reward = getPieceMoveValue(self.lastPieceMoved)

def checkDone(self):
    self.done = self.board.is_game_over() or self.numIncorrectMoves > MAX_INCORRECT_MOVES or (SOLO_PLAY and self.numMoves > MAX_MOVES)

def createObservation(self):
    obs = []
    for pos in self.positions:
        if pos is not None:
            valX = chess.square_file(pos)
            valY = chess.square_rank(pos)
        else:
            valX = -1
            valY = -1
        obs.append(valX)
        obs.append(valY)
    for pos in self.enemyPositions:
        if pos is not None:
            valX = chess.square_file(pos)
            valY = chess.square_rank(pos)
        else:
            valX = -1
            valY = -1
        obs.append(valX)
        obs.append(valY)
    return obs

def getRank(self, pieceIndex):
    return chess.square_rank(self.positions[pieceIndex])

def getFile(self, pieceIndex):
    return chess.square_file(self.positions[pieceIndex])

def moveRook(self, pieceIndex, action):
    if self.positions[pieceIndex] is None:
        return None

    if action < 8:
        return chess.square(getFile(self, pieceIndex), action)
    else:
        return chess.square(action % 8, getRank(self, pieceIndex))

def moveKnight(self, pieceIndex, action):
    if self.positions[pieceIndex] is None:
        return None

    f = getFile(self, pieceIndex)
    r = getRank(self, pieceIndex)
    if action == 0:
        f -=1
        r +=2
    elif action == 1:
        f +=1
        r +=2
    elif action == 2:
        f +=2
        r -=1
    elif action == 3:
        f +=2
        r +=1
    elif action == 4:
        f -=1
        r -=2
    elif action == 5:
        f +=1
        r -=2
    elif action == 6:
        f -=2
        r -=1
    else:
        f -=2
        r +=1
    return chess.square(f, r)


def moveBishop(self, pieceIndex, action):
    if self.positions[pieceIndex] is None:
        return None

    f = getFile(self, pieceIndex)
    r = getRank(self, pieceIndex)
    if action < 8:
        m = min(f, r)
        return chess.square(f - m + action, r - m + action)
    else:
        m = min(7 - f, r)
        return chess.square(f + m - action % 8, r - m + action % 8)

def movePawn(self, pieceIndex, action):
    if self.positions[pieceIndex] is None:
        return None
    if action < 2:
        return chess.square(getFile(self, pieceIndex), getRank(self, pieceIndex) + action +  1)
    elif action < 3:
        return chess.square(getFile(self, pieceIndex) + 1, getRank(self, pieceIndex) + 1)
    else:
        return chess.square(getFile(self, pieceIndex) - 1, getRank(self, pieceIndex) + 1)


def moveQueen(self, pieceIndex, action):
    if self.positions[pieceIndex] is None:
        return None

    f = getFile(self, pieceIndex)
    r = getRank(self, pieceIndex)
    if action < 8:
        return chess.square(getFile(self, pieceIndex), action)
    elif action < 16:
        return chess.square(action % 8, getRank(self, pieceIndex))
    elif action < 24:
        m = min(f, r)
        return chess.square(f - m + action % 8, r - m + action % 8)
    else:
        m = min(7 - f, r)
        return chess.square(f + m - action % 8, r - m + action % 8)

def moveKing(self, pieceIndex, action):
    if self.positions[pieceIndex] is None:
        return None

    f = getFile(self, pieceIndex)
    r = getRank(self, pieceIndex)
    if action == 0:
        f +=1
    elif action == 1:
        f -=1
    elif action == 2:
        r +=1
    elif action == 3:
        r -=1
    elif action == 4:
        f +=1
        r +=1
    elif action == 5:
        f +=1
        r -=1
    elif action == 6:
        f -=1
        r +=1
    else:
        f -=1
        r -=1
    return chess.square(f, r)

def resolveDestination(self, action):
    if action < 16:
        return (0, moveRook(self, 0, action))
    elif action < 24:
        return (1, moveKnight(self, 1, action % 8))
    elif action < 40:
        return (2, moveBishop(self, 2, action % 16))
    elif action < 72:
        return (3, moveQueen(self, 3, action % 32))
    elif action < 80:
        return (4, moveKing(self, 4, action % 8))
    elif action < 96:
        return (5, moveBishop(self, 5, action % 16))
    elif action < 104:
        return (6, moveKnight(self, 6, action % 8))
    elif action < 120:
        return (7, moveRook(self, 7, action % 16))
    elif action < 124:
        return (8, movePawn(self, 8, action % 4))
    elif action < 128:
        return (9, movePawn(self, 9, action % 4))
    elif action < 132:
        return (10, movePawn(self, 10, action % 4))
    elif action < 136:
        return (11, movePawn(self, 11, action % 4))
    elif action < 140:
        return (12, movePawn(self, 12, action % 4))
    elif action < 144:
        return (13, movePawn(self, 13, action % 4))
    elif action < 148:
        return (14, movePawn(self, 14, action % 4))
    elif action < 152:
        return (15,movePawn(self, 15, action % 4))
    else:
        return None

def actionToMove(self, action):
    move = None
    dest = resolveDestination(self, action)
    if dest[0] is not None:
        origSquare = self.positions[dest[0]]
        if origSquare is not None and dest[1] is not None and dest[1] > 0 and dest[1] < 64:
            move = chess.Move(origSquare, dest[1])
    return move

def processAction(self, action):
    # print(str(action) + ", "  + str(move) + ", legal=" + str(move in self.board.legal_moves))
    move = actionToMove(self, action)
    if move in self.board.legal_moves:
        self.board.push(move)
        #self.lastPieceMoved = getPiece(dest[0])
        return move
    else:
        return None

def getPiece(index):
    if index == 0 or index == 7 :
        return chess.ROOK
    elif index == 1 or index == 6:
        return chess.KNIGHT
    elif index == 2 or index == 5:
        return chess.BISHOP
    elif index == 3:
        return chess.QUEEN
    elif index == 4:
        return chess.KING
    else:
        return chess.PAWN

def updatePositions(self):
    self.positions = getPositions(self, chess.WHITE)
    self.enemyPositions = getPositions(self, chess.BLACK)

def getPositions(self, color):
    positions = []
    for i in range(16):
        positions.append(None)
    for pos in self.board.pieces(chess.ROOK, color):
        if positions[0] is not None:
            positions[7] = pos
        else:
            positions[0] = pos
    for pos in self.board.pieces(chess.KNIGHT, color):
        if positions[1] is not None:
            positions[6] = pos
        else:
            positions[1] = pos
    for pos in self.board.pieces(chess.BISHOP, color):
        if positions[2] is not None:
            positions[5] = pos
        else:
            positions[2] = pos
    for pos in self.board.pieces(chess.QUEEN, color):
        positions[3] = pos
    for pos in self.board.pieces(chess.KING, color):
        positions[4] = pos
    for i,pos in enumerate(self.board.pieces(chess.PAWN, color)):
        positions[8 + i] = pos
    return positions

def calculatePieceBalance(self):
    balance = calculatePoints(self.positions) - calculatePoints(self.enemyPositions)
    self.currentBalance = balance - self.lastPieceBalance 
    self.lastPieceBalance = balance 
    #balance = calculatePoints(self.enemyPositions)
    #self.currentBalance = self.lastPieceBalance - balance
    #self.lastPieceBalance = balance


def calculatePoints(positions):
    val = 0
    for i, pos in enumerate(positions):
        val += 0 if pos is None else calculatePieceValue(i)
    return val

def calculatePieceValue(pieceIndex):
    piece = getPiece(pieceIndex)
    if piece == chess.PAWN:
        return 2
    elif piece == chess.ROOK:
        return 5
    elif piece == chess.KNIGHT or piece == chess.BISHOP: 
        return 3
    else: 
        return 9

def getPieceMoveValue(piece):
    if piece == chess.PAWN:
        return 1
    elif piece == chess.ROOK:
        return 0.4
    elif piece == chess.KNIGHT or piece == chess.BISHOP: 
        return 0.8
    elif piece == chess.QUEEN:
        return 0.6
    else: 
        return 0.3

def manageLog(self, action):
    log = np.append(self.positions, action)
    self.logs.append(log)
    if self.numIncorrectMoves > MAX_INCORRECT_MOVES:
        print("Max incorrect moves reached:")
        for log in self.logs:
            print(log)
        print("legal moves:" + str(self.board.legal_moves))


class ChessEnv(gym.Env):

    def __init__(self):
        super(ChessEnv, self).__init__()
        self.rendering = False
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci('stockfish')
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(152)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(64,), dtype=np.float32)

    def step(self, action):
        #print(self.positions)
        #self.previousMove = self.lastMove
        self.lastMove = processAction(self, action)
        if self.lastMove is None:
            self.numIncorrectMoves += 1
        else:
            self.numMoves += 1
            self.numIncorrectMoves = 0
        self.previousStatus = getBoardStatus(self)
        if SOLO_PLAY:
            self.board.turn = chess.WHITE
        if self.board.turn == chess.BLACK: 
            result = self.engine.play(self.board, chess.engine.Limit(time=0.001))
            self.board.push(result.move)
        self.lastStatus = getBoardStatus(self)
        self.observation = createObservation(self)
        checkDone(self)
        updatePositions(self)
        calculatePieceBalance(self)
        calculateReward(self)
        #print(self.reward)
        #if self.lastMove is not None:
        #    print(self.lastPieceBalance)
        #    print(self.positions)
        #    print(self.enemyPositions)
        #    print(self.reward)
        #if self.done:
        #    print(self.reward)
        #    print(action)
        #    print("done")
        manageLog(self, action)
        return self.observation, self.reward, self.done, {}

    def reset(self):
        self.positions = [chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1, chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2, chess.H2]
        self.enemyPositions = [chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8, chess.A7, chess.B7, chess.C7, chess.D7, chess.E7, chess.F7, chess.G7, chess.H7]
        self.previousStatus = None
        self.lastMove = None
        self.previousMove = None
        self.lastPieceMoved = None
        self.lastStatus = getBoardStatus(self)
        self.observation = createObservation(self)
        self.lastPieceBalance = calculatePoints(self.enemyPositions)
        self.board.reset()
        self.numIncorrectMoves = 0
        self.numMoves = 0
        self.logs = deque([], maxlen = MAX_LOG)
        self.lastPiece = None
        return self.observation

    def initRendering(self):
        pygame.init()
        self.fpsClock = pygame.time.Clock()
        self.window_obj = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), 0, 32)
        pygame.display.set_caption("Chess")

    def render(self):
        svgBoard = chess.svg.board(self.board, size=WINDOW_SIZE)
        boardPng = cairosvg.svg2png(bytestring=str(svgBoard))
        byte_io = BytesIO(boardPng)
        img = pygame.image.load(byte_io)
        self.window_obj.blit(img, (0,0))
        pygame.display.update()

