from agent import DeepQLearningAgent
from game import SnakeGame
import time

# Training configuration
STATE_SIZE = 20  # Must match game grid size
ACTION_SIZE = 4  # Fixed: up/right/down/left
EPISODES = 1000  # Total training episodes
SAVE_INTERVAL = 100  # Save model every N episodes
MODEL_PATH = "model.pth"  # Path to save trained model

def train_agent():
    # Initialize game environment (no visualization during training)
    game = SnakeGame(state_size=STATE_SIZE, visualize=False)
    
    # Initialize Deep Q-Learning agent
    agent = DeepQLearningAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        gamma=0.95,        # Discount factor
        epsilon=1.0,       # Initial exploration rate
        epsilon_min=0.01,  # Minimum exploration rate
        epsilon_decay=0.995,  # Exploration decay rate
        lr=0.001,          # Learning rate
        batch_size=64      # Experience replay batch size
    )
    print(f"Training started with {EPISODES} episodes. Using device: {agent.device}")
    print("Episode | Total Reward | Epsilon | Training Time")
    print("-" * 60)
    
    # Track training time
    start_time = time.time()
    
    # Training loop
    for episode in range(1, EPISODES + 1):
        state = game.reset()  # Reset game to initial state
        total_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Agent selects action
            action = agent.act(state)
            
            # Execute action in game environment
            next_state, reward, done = game.step(action)
            
            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Update current state and total reward
            state = next_state
            total_reward += reward
            
            # Train network using experience replay
            agent.replay()
        
        # Calculate episode training time
        elapsed_time = time.time() - start_time
        
        # Print training progress (formatted)
        print(f"{episode:>7} | {total_reward:>12.2f} | {agent.epsilon:>7.4f} | {elapsed_time:>12.2f}s")
        
        # Save model at specified intervals
        if episode % SAVE_INTERVAL == 0:
            agent.save(MODEL_PATH)
            print(f"Model saved at episode {episode} to {MODEL_PATH}")
    
    # Save final model after all episodes
    agent.save(MODEL_PATH)
    print(f"\nTraining completed! Final model saved to {MODEL_PATH}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    train_agent()