from Agent import Agent
import numpy as np
import gym
import warnings
import torch

from utils import plotScores
warnings.filterwarnings("ignore")

env = gym.make('SpaceInvaders-v4', render_mode='human')
agent = Agent(gamma=0.75, eps=1.0, LR=0.03, max_mem_size=3000, replace=None)

while agent.mem_counter < agent.max_mem_size:
    obs = env.reset()
    done = False
    while not done:
        # 0=no action;1=fire;2=right;3=left;4=move right fire;5=move left fire
        action = env.action_space.sample()
        obs_, reward, done, info = env.step(action)
        if done and info['lives'] == 0:
            reward = -100

        agent.store_transition(np.mean(obs[15:200, 30:125], axis=2), action, reward,
                               np.mean(obs_[15:200, 30:125], axis=2))
        obs = obs_

x_axis = []
scores = []
eps_history = []
episodes = 5  # for observation

for i in range(episodes):
    eps_history.append(agent.eps)
    done = False
    obs = env.reset()
    frames = [np.sum(obs[15:200, 30:125], axis=2)]
    score = 0
    last_action = 0

    while not done:
        if len(frames) == 3:
            action = agent.choose_action(frames)
            frames = []
        else:
            action = last_action

        obs_, reward, done, info = env.step(action)
        score += reward
        frames.append(np.sum(obs_[15:200, 30:125], axis=2))
        if done and info['lives'] == 0:
            reward = -100

        agent.store_transition(np.mean(obs[15:200, 30:125], axis=2), action, reward,
                               np.mean(obs_[15:200, 30:125], axis=2))

        obs = obs_
        agent.learn(16)
        last_action = action
        # save model

    torch.save(agent.Q_eval.state_dict(), 'parameters.pth')
    x_axis.append(i+1)
    scores.append(score)
    print('Episode:', i+1, 'eps: %.4f' % agent.eps, 'Score: ', score)
    # plotScores(x_axis, scores, 'testing-plot')
    plotScores(x_axis, scores, 'result-plot')
