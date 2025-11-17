import torch
from agent import DeepQLearningAgent

# Test configuration (matches game environment)
STATE_SIZE = 20  # 20x20 grid
ACTION_SIZE = 4   # 4 possible actions (up/right/down/left)

def test_agent():
    # Initialize the agent
    agent = DeepQLearningAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    print(f"Agent initialized. Using device: {agent.device}")
    
    # Test network construction
    print(f"Q-network structure:\n{agent.model}")
    
    # Generate a random test state (20x20 grid)
    test_state = torch.rand(STATE_SIZE, STATE_SIZE).numpy()  # Simulate game state
    
    # Test action selection (should return integer between 0-3)
    action = agent.act(test_state)
    assert 0 <= action < ACTION_SIZE, "Action selection failed: invalid action value"
    print(f"Test action: {action} (valid)")
    
    # Test experience storage
    next_state = torch.rand(STATE_SIZE, STATE_SIZE).numpy()
    agent.remember(test_state, action, reward=1.0, next_state=next_state, done=False)
    assert len(agent.memory) == 1, "Experience storage failed: memory not updated"
    print("Experience stored successfully")
    
    print("All basic agent tests passed!")

if __name__ == "__main__":
    test_agent()