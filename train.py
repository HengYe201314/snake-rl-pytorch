from agent import DeepQLearningAgent
from game import SnakeGame
import time
import csv
import os
import torch

# Configuration
STATE_SIZE = 20
ACTION_SIZE = 4
EPISODES = 1000
SAVE_INTERVAL = 100
MODEL_PATH = "model.pth"
LOG_PATH = "training_log.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_agent():
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", LOG_PATH)
    
    # Initialize environment and agent
    game = SnakeGame(state_size=STATE_SIZE, visualize=False)
    agent = DeepQLearningAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        device=DEVICE
    )
    
    # Initialize log
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Episode", "Total Reward", "Epsilon", "Time (s)"])
    
    print(f"Training on {DEVICE} for {EPISODES} episodes...")
    print("Episode | Reward | Epsilon | Time")
    print("-" * 40)
    start_time = time.time()
    
    try:
        for episode in range(1, EPISODES + 1):
            state = game.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 1000:  # Prevent infinite loops
                steps += 1
                action = agent.act(state)
                next_state, reward, done = game.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # Train if enough data
                if len(agent.memory) >= agent.batch_size:
                    agent.replay()
            
            # Log and print progress
            elapsed = time.time() - start_time
            print(f"{episode:>7} | {total_reward:>6.1f} | {agent.epsilon:.3f} | {elapsed:.1f}s")
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([episode, round(total_reward, 1), round(agent.epsilon, 3), round(elapsed, 1)])
            
            # Save model
            if episode % SAVE_INTERVAL == 0:
                agent.save(MODEL_PATH)
                print(f"Model saved at episode {episode}")
    
    except Exception as e:
        print(f"Error: {str(e)}. Saving emergency model...")
        agent.save(f"emergency_model_{episode}.pth")
    
    # Final save
    agent.save(MODEL_PATH)
    print(f"\nTraining complete. Model saved to {MODEL_PATH}")
    print(f"Logs saved to {log_file}")

if __name__ == "__main__":
    train_agent()