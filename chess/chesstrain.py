from stable_baselines3 import PPO
from torch.utils.data.dataset import Dataset, random_split
from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from chessenv import ChessEnv, actionToMove
import time, os
import chess
import chess.svg
import chess.engine
from stockfish import Stockfish

trainings_dir = f"trainings/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(trainings_dir):
    os.makedirs(trainings_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

def moveRook(self, pieceIndex, origPos, destPos):
    action = 0 if pieceIndex == 0 else 8
    origFile = chess.square_file(origPos)
    origRank = chess.square_rank(origPos)
    destFile = chess.square_file(destPos)
    destRank = chess.square_rank(destPos)
    offset = pieceActionOffset(pieceIndex)
    if origFile == destFile:
        return offset + destRank
    else :
        return offset + 8 + destFile

def moveKnight(self, pieceIndex, origPos, destPos):
    origFile = chess.square_file(origPos)
    origRank = chess.square_rank(origPos)
    destFile = chess.square_file(destPos)
    destRank = chess.square_rank(destPos)
    offset = pieceActionOffset(pieceIndex)
    if destFile - origFile == -1 and destRank - origRank == 2: 
        return offset + 0
    elif destFile - origFile == 1 and destRank - origRank == 2: 
        return offset + 1
    elif destFile - origFile == 2 and destRank - origRank == -1: 
        return offset + 2
    elif destFile - origFile == 2 and destRank - origRank == 1: 
        return offset + 3
    elif destFile - origFile == -1 and destRank - origRank == -2: 
        return offset + 4
    elif destFile - origFile == 1 and destRank - origRank == -2: 
        return offset + 5
    elif destFile - origFile == -2 and destRank - origRank == -1: 
        return offset + 6
    elif destFile - origFile == -2 and destRank - origRank == 1: 
        return offset + 7
    else:
        return None


def moveBishop(self, pieceIndex, origPos, destPos):
    origFile = chess.square_file(origPos)
    origRank = chess.square_rank(origPos)
    destFile = chess.square_file(destPos)
    destRank = chess.square_rank(destPos)
    offset = pieceActionOffset(pieceIndex)
    if destFile - origFile > 0:
        return offset + min(destFile, destRank)
    else:
        return offset + 8 + min(7 - destFile, destRank)


def movePawn(self, pieceIndex, origPos, destPos):
    origFile = chess.square_file(origPos)
    origRank = chess.square_rank(origPos)
    destFile = chess.square_file(destPos)
    destRank = chess.square_rank(destPos)
    offset = pieceActionOffset(pieceIndex)
    if origFile == destFile:
        return offset + destRank - origRank - 1
    else :
        if destFile - origFile > 0:
            return offset + 2
        else :
            return offset + 3

def moveQueen(self, pieceIndex, origPos, destPos):
    origFile = chess.square_file(origPos)
    origRank = chess.square_rank(origPos)
    destFile = chess.square_file(destPos)
    destRank = chess.square_rank(destPos)
    offset = pieceActionOffset(pieceIndex)

    if origFile == destFile:
        return offset + destRank
    elif origRank == destRank :
        return offset + 8 + destFile
    elif destFile - origFile > 0:
        return offset + 16 + min(destFile, destRank)
    else:
        return offset + 24 + min(7 - destFile, destRank)

def moveKing(self, pieceIndex, origPos, destPos):
    origFile = chess.square_file(origPos)
    origRank = chess.square_rank(origPos)
    destFile = chess.square_file(destPos)
    destRank = chess.square_rank(destPos)
    offset = pieceActionOffset(pieceIndex)

    if destFile - origFile == 1 and destRank - origRank == 0: 
        return offset + 0
    elif destFile - origFile == -1 and destRank - origRank == 0: 
        return offset + 1
    elif destFile - origFile == 0 and destRank - origRank == 1: 
        return offset + 2
    elif destFile - origFile == 0 and destRank - origRank == -1: 
        return offset + 3
    elif destFile - origFile == 1 and destRank - origRank == 1: 
        return offset + 4
    elif destFile - origFile == 1 and destRank - origRank == -1: 
        return offset + 5
    elif destFile - origFile == -1 and destRank - origRank == 1: 
        return offset + 6
    elif destFile - origFile == -1 and destRank - origRank == -1: 
        return offset + 7
    else:
        return None

def moveToAction(env, move):
    origPos = move.from_square
    destPos = move.to_square
    pieceIndex = getPieceIndex(env, origPos)
    if pieceIndex == 0:
        return moveRook(env, 0, origPos, destPos)
    elif pieceIndex == 1:
        return moveKnight(env, 1, origPos, destPos)
    elif pieceIndex == 2:
        return moveBishop(env, 2, origPos, destPos)
    elif pieceIndex == 3:
        return moveQueen(env, 3, origPos, destPos)
    elif pieceIndex == 4:
        return moveKing(env, 4, origPos, destPos)
    elif pieceIndex == 5:
        return moveBishop(env, 5, origPos, destPos)
    elif pieceIndex == 6:
        return moveKnight(env, 6, origPos, destPos)
    elif pieceIndex == 7:
        return moveRook(env, 7, origPos, destPos)
    elif pieceIndex == 8:
        return movePawn(env, 8, origPos, destPos)
    elif pieceIndex == 9:
        return movePawn(env, 9, origPos, destPos)
    elif pieceIndex == 10:
        return movePawn(env, 10, origPos, destPos)
    elif pieceIndex == 11:
        return movePawn(env, 11, origPos, destPos)
    elif pieceIndex == 12:
        return movePawn(env, 12, origPos, destPos)
    elif pieceIndex == 13:
        return movePawn(env, 13, origPos, destPos)
    elif pieceIndex == 14:
        return movePawn(env, 14, origPos, destPos)
    elif pieceIndex == 15:
        return movePawn(env, 15, origPos, destPos)
    else:
        return None


def pieceActionOffset(pieceIndex):
    if pieceIndex == 0:
        return 0
    elif pieceIndex == 1:
        return 16
    elif pieceIndex == 2:
        return 24
    elif pieceIndex == 3:
        return 40
    elif pieceIndex == 4:
        return 72
    elif pieceIndex == 5:
        return 80
    elif pieceIndex == 6:
        return 96
    elif pieceIndex == 7:
        return 104
    elif pieceIndex == 8:
        return 120
    elif pieceIndex == 9:
        return 124
    elif pieceIndex == 10:
        return 128
    elif pieceIndex == 11:
        return 132
    elif pieceIndex == 12:
        return 136
    elif pieceIndex == 13:
        return 140
    elif pieceIndex == 14:
        return 144
    elif pieceIndex == 15:
        return 148
    else:
        return None

def getPieceIndex(env, square):
    for i,pos in enumerate(env.positions):
        if pos == square:
            return i
    return None

def pretrain_student(
        student,
        test_dataset,
        batch_size=64,
        epochs=1000,
        scheduler_gamma=0.7,
        learning_rate=1.0,
        log_interval=100,
        no_cuda=True,
        seed=1,
        test_batch_size=64,
    ):
    use_cuda = not no_cuda and th.cuda.is_available()
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    criterion = nn.MSELoss()

    # Extract initial policy
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            action, _, _ = model(data)
            action_prediction = action.double()
            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                action, _, _ = model(data)
                action_prediction = action.double()
                test_loss = criterion(action_prediction, target)
            test_loss /= len(test_loader.dataset)
            print(f"Test set: Average loss: {test_loss:.4f}")

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = th.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = th.utils.data.DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=True, **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student.policy = model

env = ChessEnv()
num_interactions = int(4e4)
expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
expert_actions = np.empty((num_interactions,) + env.action_space.shape)

obs = env.reset()

engine = chess.engine.SimpleEngine.popen_uci('stockfish')
board = chess.Board()
board.castling_rights = 0


for i in tqdm(range(num_interactions)):
    result = env.engine.play(env.board, chess.engine.Limit(time=0.001))
    action = moveToAction(env, result.move)
    if action is None:
        raise RuntimeError("unknonw action for move:" + str(result.move) + str(env.positions))

    dest = actionToMove(env, action) 
    if dest != result.move:
        raise RuntimeError("incorrect move for :" + str(result.move) + ",action = " + str(action) + ", estimated = " + str(dest) + str(env.positions))

    env.board.push(result.move)

    expert_observations[i] = obs
    expert_actions[i] = action
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

np.savez_compressed(
        "expert_data",
        expert_actions=expert_actions,
        expert_observations=expert_observations,
        )

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)

expert_dataset = ExpertDataSet(expert_observations, expert_actions)

train_size = int(0.8 * len(expert_dataset))

test_size = len(expert_dataset) - train_size

train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
        )

agent = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

pretrain_agent(
        agent,
        train_dataset=train_dataset,
        test_dataset=train_dataset,	
        epochs=3,
        scheduler_gamma=0.7,
        learning_rate=1.0,
        log_interval=100,
        no_cuda=True,
        seed=1,
        batch_size=64,
        test_batch_size=1000,
        )



