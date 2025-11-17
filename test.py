import pygame
from agent import DeepQLearningAgent
from game import SnakeGame

# Configuration (must match training parameters)
STATE_SIZE = 20
ACTION_SIZE = 4
MODEL_PATH = "model.pth"  # Path to trained model

def test_trained_model():
    # Initialize agent and load trained model
    agent = DeepQLearningAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    try:
        agent.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except:
        print(f"Model not found at {MODEL_PATH}. Using test_model.pth instead.")
        agent.load("test_model.pth")  # Fallback to test model
    
    # Disable exploration (use greedy policy)
    agent.epsilon = 0.0
    
    # Initialize game with visualization
    game = SnakeGame(state_size=STATE_SIZE, visualize=True)
    state = game.reset()
    done = False
    total_reward = 0
    
    print("Visualizing trained agent... (Close window to exit)")
    print("Controls: None (agent acts autonomously)")
    
    # Run the game with the trained agent
    while not done:
        # Handle window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
        
        if done:
            break
        
        # Agent selects action (greedy)
        action = agent.act(state)
        
        # Execute action in game
        next_state, reward, done = game.step(action)
        
        # Update state and reward
        state = next_state
        total_reward += reward
        
        # Control visualization speed (lower = slower)
        pygame.time.Clock().tick(10)
    
    # Print final results
    print(f"Test completed! Total Reward: {total_reward:.2f}")
    print(f"Final Score: {game.score}")
    game.close()

if __name__ == "__main__":
    test_trained_model()