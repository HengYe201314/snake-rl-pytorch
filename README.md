# Deep Q-Learning Snake Game: PyTorch Implementation
This project converts a TensorFlow-based DQN (Deep Q-Learning) Snake game agent to PyTorch, and trains an autonomous AI agent capable of playing the Snake game with obstacle avoidance and food-seeking behavior.

## Project Overview
- **Goal**: Implement a DQN agent to play Snake autonomously, with PyTorch as the deep learning framework.
- **Training**: 1000 episodes of training, with reward logging and model checkpointing.
- **Visualization**: Pygame-based game environment for real-time demonstration of the trained agent.

## Project Structure
| File Path               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `agent.py`              | Core DQN agent implementation (network architecture, experience replay, training logic). |
| `game.py`               | Snake game environment (Pygame visualization, state/action handling).       |
| `train.py`              | Training script: runs 1000 episodes, saves training logs and final model.   |
| `test.py`               | Demonstration script: loads trained model, runs autonomous agent (continuous window). |
| `model.pth`             | Final trained DQN model (output after 1000 training episodes).              |
| `logs/training_log.csv` | Training logs: records episode, total reward, epsilon, and step count.      |
| `requirements.txt`      | Dependencies list for environment setup.                                    |

## How to Reproduce the Results
### 1. Clone the Repository
```bash
2. Install Dependencies
pip install -r requirements.txt
3. Run the Trained Agent (Demonstration) 
No need to retrain—use the pre-trained model.pth:
python test.py
The game window will launch and run continuously (restarts automatically after collision). 
Close the game window manually to stop the demonstration.
4. Retrain the Agent (Optional) 
To re-run the training process (1000 episodes):
python train.py
Training logs will be saved to logs/training_log.csv. 
The final model will be saved as model.pth.
Key Training Results 
• Total Training Episodes: 1000 
• Average Total Reward per Episode: 22.7 
• Highest Single-Episode Reward: 38.5 
• Maximum Score (Food Eaten): 9 
• Agent Capabilities: Autonomous obstacle avoidance (walls/self-body) + active food-seeking.
Demonstration Video
The demonstration video shows the trained agent playing Snake autonomously. It includes:
Project file overview 
Execution of test.py 
Real-time agent performance (food-seeking + obstacle avoidance)
Terminal output of reward/step metrics
Technical Details
Framework: PyTorch 2.0+ 
Game Visualization: Pygame 2.5+ 
State Size: 20 (custom feature representation of the game grid) 
Action Size: 4 (up/down/left/right) 
Epsilon Decay: Linear decay from initial value to 0.0 (no random exploration in test mode)