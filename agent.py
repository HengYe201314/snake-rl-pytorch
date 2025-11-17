import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DeepQLearningAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64):
        """
        Initialize Deep Q-Learning Agent
        :param state_size: Dimension of state (grid size, e.g., 20 for 20x20 grid)
        :param action_size: Number of possible actions (4 for up/right/down/left)
        :param gamma: Discount factor for future rewards
        :param epsilon: Exploration rate (initial)
        :param epsilon_min: Minimum exploration rate
        :param epsilon_decay: Decay rate for epsilon
        :param lr: Learning rate for optimizer
        :param batch_size: Number of experiences to sample for replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay buffer (stores (s, a, r, s', done))
        self.memory = deque(maxlen=2000)
        
        # Device configuration (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build Q-network and move to device
        self.model = self._build_model().to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _build_model(self):
        """
        Build convolutional neural network for Q-value approximation
        Input: (batch_size, 1, state_size, state_size) (grayscale game grid)
        Output: Q-values for each action (batch_size, action_size)
        """
        model = nn.Sequential(
            # First convolutional layer: 1 input channel → 32 output channels
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces size by half
            
            # Second convolutional layer: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces size by half again
            
            # Flatten for fully connected layers
            nn.Flatten(),
            
            # Fully connected layer
            nn.Linear(64 * (self.state_size // 4) * (self.state_size // 4), 256),
            nn.ReLU(),
            
            # Output layer: Q-values for each action
            nn.Linear(256, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        :param state: Current state (state_size x state_size grid)
        :param action: Taken action (0-3)
        :param reward: Received reward
        :param next_state: Next state after action
        :param done: Whether episode ended (True/False)
        """
        # Convert to PyTorch tensors and add channel dimension (1 for grayscale)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Shape: (1, state_size, state_size)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose action using epsilon-greedy policy
        :param state: Current state (state_size x state_size grid)
        :return: Selected action (0-3)
        """
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        # Greedy action (exploit learned policy)
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():  # No gradient calculation for inference
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # Add batch and channel dims
            q_values = self.model(state_tensor)
        self.model.train()  # Reset to training mode
        return torch.argmax(q_values).item()  # Return action with highest Q-value

    def replay(self):
        """
        Train network using randomly sampled experiences from replay buffer
        """
        # Wait until buffer has enough experiences
        if len(self.memory) < self.batch_size:
            return
        
        # Randomly sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and concatenate into batches
        states = torch.cat(states).unsqueeze(1).to(self.device)  # Add channel dim: (batch_size, 1, state_size, state_size)
        next_states = torch.cat(next_states).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # (batch_size, 1)
        rewards = torch.FloatTensor(rewards).to(self.device)  # (batch_size,)
        dones = torch.FloatTensor(dones).to(self.device)  # (batch_size,)
        
        # Calculate current Q-values (Q(s, a))
        current_q = self.model(states).gather(1, actions).squeeze(1)  # (batch_size,)
        
        # Calculate target Q-values (r + γ * maxQ(s', a') if not done, else r)
        next_q = self.model(next_states).max(1)[0].detach()  # (batch_size,)
        target_q = rewards + (self.gamma * next_q * (1 - dones))  # (batch_size,)
        
        # Compute loss and backpropagate
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update weights
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """
        Save trained model weights
        :param path: File path to save (e.g., "model.pth")
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Load trained model weights
        :param path: File path to load from (e.g., "model.pth")
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))