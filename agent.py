import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

# Training hyperparameters (consistent with train.py)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
GAMMA = 0.95  # Discount factor for future rewards
BATCH_SIZE = 64
MEMORY_SIZE = 10000

class DeepQLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Epsilon-greedy exploration parameters
        self.epsilon = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        
        # Critical fix: Add gamma attribute (missing in previous version)
        self.gamma = GAMMA  # Required for Bellman equation in replay()
        
        # Experience replay buffer
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        # DQN Network architecture
        self.model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        self.model.train()  # Set to training mode

    def remember(self, state, action, reward, next_state, done):
        """Store experience tuple in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Select action using epsilon-greedy policy"""
        # Random exploration
        if random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        # Greedy action (using trained model)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        """Train model using random batch from replay buffer"""
        if len(self.memory) < BATCH_SIZE:
            return  # Wait until buffer has enough samples
        
        # Sample random batch from memory
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to PyTorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Calculate current Q-values
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        # Calculate target Q-values (Bellman equation)
        next_q = self.model(next_states).max(1)[0]
        target_q = rewards + (self.gamma * next_q * (1 - dones))  # Fixed: uses self.gamma
        
        # Update model
        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q)
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        """Load trained model weights from file"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # Switch to evaluation mode
        print(f"Model loaded successfully from {path}")

    def save(self, path):
        """Save current model weights to file"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")