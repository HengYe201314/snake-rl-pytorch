\# Deep Q-Learning for Snake Game (PyTorch Implementation)



This repository contains a PyTorch implementation of a Deep Q-Learning agent that learns to play the Snake game. The project is developed as a graded assignment for the Deep Learning course.



\## Project Structure

snake-rl-pytorch/

├── agent.py # Core DQN agent implementation (PyTorch)

├── game.py # Snake game environment with visualization

├── train.py # Script to train the DQN agent

├── test.py # Script to test and visualize the trained agent

├── requirements.txt # Dependencies list

├── test\_agent\_basic.py # Basic unit tests for the agent

├── test\_train.py # Test script for training flow

└── README.md # Project documentation



\## Dependencies

\- Python 3.8+

\- PyTorch 1.13.0+

\- NumPy 1.21.0+

\- Pygame 2.1.0+



Install dependencies using:

```bash

pip install -r requirements.txt

Training the Agent To train the DQN agent from scratch, run:

python train.py

Training runs for 1000 episodes by default (configurable in train.py) 

The model is saved every 100 episodes as model.pth 

Testing the Trained Agent 

To visualize the trained agent's performance, run:Training progress is printed to the console (episode number, total reward, epsilon, time)

python test.py

This loads model.pth by default (falls back to test\\\_model.pth if not found) 

A window will open showing the snake moving autonomously

Exploration is disabled (epsilon=0) to use the learned policy

Key Files Explanation

agent.py: Implements the Deep Q-Learning agent with experience replay and epsilon-greedy policy. The Q-network is a convolutional neural network (CNN) that takes the game grid as input. 

game.py: Implements the Snake game environment with state representation (20x20 grid) and reward mechanism (positive reward for eating food, negative reward for collision). 

train.py: Handles the training loop, including episode management, experience collection, and network updates. 

test.py: Loads a trained model and visualizes its performance in the game environment.

Trained Model

The final trained model (model.pth) is available at: \\\[Google Drive Link] (to be filled after full training)

Author

HengYe Ding


