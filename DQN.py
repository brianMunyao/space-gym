import torch.nn as nn
import torch as T
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, LR):
        super(DQN, self).__init__()
        # use conv to reduce state space
        # 1 input channel (color doesn't matter - save computation)
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.fc1 = nn.Linear(128*19*8, 512)
        # 6 actions (L,R,shoot static, shoot while moving left, shoot while moving right,skip)
        self.fc2 = nn.Linear(512, 6)

        self.optimizer = optim.RMSprop(self.parameters(), lr=LR)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        # convert sequence of frames to Tensor
        obs = T.Tensor(observation).to(self.device)
        obs = obs.view(-1, 1, 185, 95)  # reshape for conv layer
        obs = F.relu(self.conv1(obs))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))

        # flatten convolved images; then feed into fc
        obs = obs.view(-1, 128*19*8)
        obs = F.relu(self.fc1(obs))

        actions = self.fc2(obs)

        return actions  # this will be a matrix: k x 6 where k=num imgs passed in
