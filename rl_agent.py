import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 actions: EDGE / CLOUD
        )

    def forward(self, x):
        return self.net(x)


class RLAgent:
    def __init__(self, state_dim, device):
        self.device = device
        self.model = DQN(state_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        # self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        # self.epsilon_decay = 0.98
        self.epsilon_min = 0.05

    def select_action(self, state):
        if random.random() < self.epsilon:
            # return torch.randint(0, 2, (state.size(0),))
            return torch.randint(0, 2, (state.size(0),)).to(self.device)

        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values, dim=1)

    def store(self, transition):
        self.memory.append(transition)

    # def train(self, batch_size=64):
    #     if len(self.memory) < batch_size:
    #         return

    #     batch = random.sample(self.memory, batch_size)

    #     states, actions, rewards, next_states = zip(*batch)

    #     states = torch.cat(states).to(self.device)
    #     actions = torch.cat(actions).to(self.device)
    #     rewards = torch.cat(rewards).float().to(self.device)
    #     next_states = torch.cat(next_states).to(self.device)

    #     q_values = self.model(states)
    #     q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    #     with torch.no_grad():
    #         next_q = self.model(next_states).max(1)[0]

    #     # target = rewards + self.gamma * next_q
    #     target = rewards
    #     # target = rewards + 0.5 * next_q


    #     loss = nn.MSELoss()(q_values, target)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     # decay epsilon
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        states, actions, rewards = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).float().to(self.device)

        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # BANDIT → target = reward (no next state, no gamma)
        target = rewards.detach()

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
