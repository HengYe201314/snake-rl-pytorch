import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DeepQLearningAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, 
                 batch_size=64, device=None):
        self.state_size = state_size  # 20
        self.action_size = action_size  # 4
        self.memory = deque(maxlen=2000)  # Replay buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        
        # Device configuration
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Q-Network (input: 1x20x20)
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 32x10x10
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 64x5x5
            nn.Flatten(),     # Output: 64*5*5=1600
            nn.Linear(1600, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # Store as (state, action, reward, next_state, done) with correct shape
        state_tensor = torch.FloatTensor(state).to(self.device)  # (1,20,20)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)  # (1,20,20)
        self.memory.append((state_tensor, action, reward, next_state_tensor, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        
        # Exploit: use model to predict best action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1,1,20,20)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Unpack and prepare tensors
        states = torch.cat([s.unsqueeze(0) for s, _, _, _, _ in batch])  # (64,1,20,20)
        actions = torch.tensor([a for _, a, _, _, _ in batch], device=self.device)  # (64,)
        rewards = torch.tensor([r for _, _, r, _, _ in batch], device=self.device, dtype=torch.float32)  # (64,)
        next_states = torch.cat([ns.unsqueeze(0) for _, _, _, ns, _ in batch])  # (64,1,20,20)
        dones = torch.tensor([d for _, _, _, _, d in batch], device=self.device, dtype=torch.float32)  # (64,)
        
        # Calculate Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # (64,)
        next_q = self.model(next_states).max(1)[0]  # (64,)
        target_q = rewards + (self.gamma * next_q * (1 - dones))  # (64,)
        
        # Update model
        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q.detach())
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))