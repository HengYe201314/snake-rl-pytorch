from agent import DeepQLearningAgent
from game import SnakeGame
import pygame
import time
import os  # Import here to avoid circular issues

# Configuration
STATE_SIZE = 20
ACTION_SIZE = 4
MODEL_PATH = "model.pth"  # Path to trained model

def test_agent():
    # Initialize game with visualization
    game = SnakeGame(state_size=STATE_SIZE, visualize=True)
    # Initialize agent and load model
    agent = DeepQLearningAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found! Train first.")
        game.close()
        return
    
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0  # Disable exploration (use trained policy)

    print("=" * 50)
    print("Trained Agent Demonstration - Snake Game")
    print("=" * 50)
    print(f"Loaded model from: {MODEL_PATH}")
    print("Close the game window to exit.")
    print("=" * 50)

    # Continuous demonstration loop
    while True:
        state = game.reset()
        done = False
        total_reward = 0
        episode_steps = 0
        food_eaten = 0

        while not done:
            # Handle window close
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.close()
                    print("\nDemonstration ended. Exiting...")
                    return

            # Agent takes action
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            
            # Update metrics
            total_reward += reward
            episode_steps += 1
            if game.food_eaten:
                food_eaten += 1
            
            state = next_state

        # Print episode stats and restart
        print(f"Episode ended | Reward: {total_reward:.2f} | Steps: {episode_steps} | Food: {food_eaten} | Restarting in 1s...")
        time.sleep(1)

if __name__ == "__main__":
    test_agent()