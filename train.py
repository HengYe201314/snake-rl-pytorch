from agent import DeepQLearningAgent
from game import SnakeGame
import os
import csv
from datetime import datetime

# Training configuration
STATE_SIZE = 20
ACTION_SIZE = 4
EPISODES = 1500  # Total training episodes
MAX_STEPS_PER_EPISODE = 300
MODEL_SAVE_PATH = "model.pth"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

def main():
    # Create log directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # Initialize game and agent
    game = SnakeGame(state_size=STATE_SIZE, visualize=False)  # No visualization during training
    agent = DeepQLearningAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    
    # Initialize log file with header
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total_Reward", "Epsilon", "Total_Steps", "Food_Eaten", "Collision", "Timestamp"])
    
    print("=" * 60)
    print("DQN Snake Game Training (PyTorch)")
    print(f"Total Episodes: {EPISODES} | State Size: {STATE_SIZE} | Action Size: {ACTION_SIZE}")
    print("=" * 60)
    
    # Main training loop
    for episode in range(1, EPISODES + 1):
        state = game.reset()
        total_reward = 0.0
        total_steps = 0
        food_eaten_count = 0
        collision = False
        
        while total_steps < MAX_STEPS_PER_EPISODE and not collision:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            
            # Update metrics
            total_reward += reward
            total_steps += 1
            if game.food_eaten:
                food_eaten_count += 1
            if game.collision:
                collision = True
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train model
            agent.replay()
            
            state = next_state
        
        # Save model periodically
        if episode % 100 == 0 or episode == EPISODES:
            agent.save(MODEL_SAVE_PATH)
        
        # Log episode data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                episode, round(total_reward, 2), round(agent.epsilon, 4),
                total_steps, food_eaten_count, collision, timestamp
            ])
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode: {episode:4d} | Reward: {total_reward:6.2f} | Epsilon: {agent.epsilon:.4f} | Steps: {total_steps:3d} | Food: {food_eaten_count} | Collision: {collision}")
    
    # Final cleanup
    agent.save(MODEL_SAVE_PATH)
    game.close()
    print("=" * 60)
    print("Training Completed!")
    print(f"Final model saved to: {MODEL_SAVE_PATH}")
    print(f"Training logs saved to: {LOG_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()