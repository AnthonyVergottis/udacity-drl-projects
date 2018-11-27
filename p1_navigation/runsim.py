import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from dqnAgent import Agent

env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

agent = Agent(state_size=8, action_size=4, seed=0)

def DQN(n_episodes=2000,max_t=1000,eps_start=1.0,eps_end=0.01,eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1,n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.getAction(state,eps)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            agent.step(state,action,reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.networkLocal.state_dict(), 'checkpoint.pth')
            break
    return scores

dqn = DQN()
