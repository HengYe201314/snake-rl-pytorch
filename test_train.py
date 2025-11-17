from agent import DeepQLearningAgent
from game import SnakeGame
import time

# Test configuration (short training run for validation)
STATE_SIZE = 20
ACTION_SIZE = 4
TEST_EPISODES = 5  # Only 5 episodes for testing
TEST_MODEL_PATH = "test_model.pth"  # Temporary model path

def test_training_flow():
    # Initialize game environment and agent
    game = SnakeGame(state_size=STATE_SIZE, visualize=False)
    agent = DeepQLearningAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        batch_size=32  # Smaller batch size for faster test
    )
    
    print("=== Training Flow Test (5 Episodes) ===")
    print("This test verifies if the training loop runs without errors.")
    print("Episode | Total Reward | Epsilon | Time (s)")
    print("-" * 50)
    
    start_time = time.time()
    for episode in range(1, TEST_EPISODES + 1):
        state = game.reset()  # Reset game to initial state
        total_reward = 0
        done = False
        
        # Episode loop
        while not done:
            action = agent.act(state)  # Agent selects action
            next_state, reward, done = game.step(action)  # Execute action
            agent.remember(state, action, reward, next_state, done)  # Store experience
            agent.replay()  # Train using experience replay
            state = next_state  # Update state
            total_reward += reward  # Accumulate reward
        
        # Calculate elapsed time and print progress
        elapsed_time = time.time() - start_time
        print(f"{episode:>7} | {total_reward:>12.2f} | {agent.epsilon:>7.4f} | {elapsed_time:>6.2f}")
    
    # Save temporary test model
    agent.save(TEST_MODEL_PATH)
    print("\nTest completed! Temporary model saved as 'test_model.pth'")
    print("Training flow is valid if no errors occurred.")

if __name__ == "__main__":
    test_training_flow()