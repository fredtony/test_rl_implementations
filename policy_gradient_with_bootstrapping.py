"""Implementing 'vanilla' policy gradient as described in Schulman's thesis.

The thesis can be found at
https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-217.html
Code derived from
https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
The main difference is adding a second net to learn to approximate the value function.
"""
from typing import List
import warnings

import numpy as np
import gym
from numpy.lib.financial import _npv_dispatcher
import torch
from torch import nn, optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

GAMMA = 0.99
EPS = np.finfo(np.float32).eps.item()
MAX_STEPS = 1000
MAX_EPISODES = 3000
SHOW_RENDER = False
LOG_INTERVAL = 10
LR = 1e-2
GAME = "LunarLander-v2"  # "CartPole-v1"
N_INPUTS = 8  # 4
N_HIDDEN = 512  # 256
N_OUTPUTS = 4  # 2

policy_net = nn.Sequential(
    nn.Linear(N_INPUTS, N_HIDDEN),
    nn.Dropout(p=0.6),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, N_OUTPUTS),
    nn.Softmax(dim=1),
)

value_net = nn.Sequential(
    nn.Linear(N_INPUTS, N_HIDDEN), nn.Dropout(p=0.6), nn.ReLU(), nn.Linear(N_HIDDEN, 1),
)


class MyAgent:
    def __init__(self, n_inputs: int = 4, n_hidden: int = 256, n_outputs: int = 2):
        self.policy = policy_net
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.value_func = value_net
        self.value_func_optimizer = optim.Adam(self.value_func.parameters(), lr=LR)
        # Saved values from episodes
        self.episode_states: List[torch.Tensor] = []
        self.episode_rewards: List[float] = []
        self.episode_log_probs: List[torch.Tensor] = []

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.episode_states.append(state)
        prob_dist = Categorical(self.policy.forward(state))
        action = prob_dist.sample()
        self.episode_log_probs.append(prob_dist.log_prob(action))
        return action.item()

    def run_episode(self, env, render=False):
        state = env.reset()
        time_step = 0
        for time_step in range(MAX_STEPS):
            action = self.act(state)
            state, reward, done, _ = env.step(action)
            if render or SHOW_RENDER:
                env.render()
            self.episode_rewards.append(reward)
            if done:
                break
        return sum(self.episode_rewards), time_step + 1

    def train_policy_net(self, policy_loss):
        self.policy_optimizer.zero_grad()
        policy_loss = policy_loss.sum()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

    def train_value_net(self, value_loss):
        self.value_func_optimizer.zero_grad()
        value_loss = value_loss.pow(2).sum()
        value_loss.backward(retain_graph=True)
        self.value_func_optimizer.step()

    def train_on_episode(self):
        adv_ests = []
        policy_loss = np.zeros(len(self.episode_rewards), dtype=np.float32)
        R = 0
        for i in range(len(self.episode_rewards) - 1):
            R = self.episode_rewards[i]
            V_current = self.value_func.forward(self.episode_states[i]).squeeze(dim=1)
            V_next = self.value_func.forward(self.episode_states[i + 1]).squeeze(dim=1)
            adv_ests.append(R + GAMMA * V_next - V_current)
        adv_ests.append(self.episode_rewards[-1] - V_current)
        adv_ests = torch.cat(adv_ests)
        # print(adv_ests)
        # print(-torch.cat(self.episode_log_probs))
        self.train_value_net(adv_ests)
        # adv_ests = (adv_ests - adv_ests.mean()) / (adv_ests.std() + EPS)
        policy_loss = -torch.cat(self.episode_log_probs) * adv_ests
        self.train_policy_net(policy_loss)
        self.episode_states.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()


agent = MyAgent()
env = gym.make(GAME)

running_reward = np.zeros(MAX_EPISODES)
episode_rewards = np.zeros(MAX_EPISODES)
reward_decay = 0.95
reward_decay_inv = 1 - reward_decay

for i in range(1, MAX_EPISODES + 1):
    state = env.reset()
    episode_rewards[i - 1], episode_length = agent.run_episode(env, render=False)
    agent.train_on_episode()
    running_reward[i] = (
        reward_decay * running_reward[i - 1] + reward_decay_inv * episode_rewards[i - 1]
    )

    if i % LOG_INTERVAL == 0:
        print(
            f"Episode {i}\tLast reward: {episode_rewards[i-1]:.2f}\t"
            f"Running reward: {running_reward[i]:.2f}"
        )
    if running_reward[i] > env.spec.reward_threshold:
        print(
            f"Solved! Running reward is now {running_reward[i]} and the last episode "
            f"runs to {episode_length} time steps!"
        )
        break

print("Running the trained agent several more times with rendering to see results.")
for _ in range(5):
    state = env.reset()
    agent.run_episode(env, render=True)

env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(i)), episode_rewards[:i])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(i)), running_reward[:i])

plt.show()
