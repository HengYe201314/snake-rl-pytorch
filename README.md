# Snake Game with DQN (PyTorch)
A Deep Q-Learning agent trained to play the Snake game, implemented with PyTorch.

## Dependencies
Install required packages first:
```bash
pip install pygame torch numpy
Train the agent:
python train.py
Training will run for 1500 episodes, with logs saved to logs/training_log.csv.
Model will be saved as model.pth every 100 episodes.
Test the trained agent:
python test.py
Requires a trained model (model.pth) generated from training. 
A Pygame window will open to show the agent playing.
Files 
• agent.py: DQN agent implementation (neural network + training logic).
• game.py: Snake game environment with reward system. 
• train.py: Training loop to train the agent. 
• test.py: Visualize the trained agent's performance.