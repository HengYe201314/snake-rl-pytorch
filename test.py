from agent import DeepQLearningAgent
from game import SnakeGame
import pygame

# Configuration (match training)
STATE_SIZE = 20
ACTION_SIZE = 4
MODEL_PATH = "model.pth"  # 确保是训练好的模型

def test_agent():
    # Initialize game (enable visualization)
    game = SnakeGame(state_size=STATE_SIZE, visualize=True)
    # Initialize agent and load trained model
    agent = DeepQLearningAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    agent.load(MODEL_PATH)
    print(f"Loaded trained model from {MODEL_PATH}")

    # Reset game to initial state
    state = game.reset()
    done = False
    total_reward = 0

    # Run until game ends (or window closed)
    while not done:
        # Handle window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
        if done:
            break

        # Agent selects action (no exploration, only exploit)
        agent.epsilon = 0.0  # 强制智能体使用训练好的策略，不随机探索
        action = agent.act(state)
        
        # Execute action
        next_state, reward, done = game.step(action)
        total_reward += reward
        state = next_state

    # Print result
    print(f"Test completed! Total Reward: {total_reward:.2f}")
    game.close()

if __name__ == "__main__":
    test_agent()