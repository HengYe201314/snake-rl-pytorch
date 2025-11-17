import pygame
import numpy as np

# Game constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLOCK_SIZE = 20  # Size of each grid block
SPEED = 100  # Game speed (for testing; auto-accelerated during training)

class SnakeGame:
    def __init__(self, state_size=20, visualize=False):
        self.state_size = state_size  # Game grid size (state_size x state_size)
        self.visualize = visualize  # Whether to enable visualization
        self.width = self.state_size * BLOCK_SIZE
        self.height = self.state_size * BLOCK_SIZE
        
        # Initialize Pygame (only if visualization is enabled)
        if self.visualize:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake RL (PyTorch)')
            self.clock = pygame.time.Clock()

        self.reset()  # Reset game state on initialization
        return

    def reset(self):
        # Initialize snake position (center) and direction
        self.snake = [
            (self.state_size // 2, self.state_size // 2),
            (self.state_size // 2 - 1, self.state_size // 2),
            (self.state_size // 2 - 2, self.state_size // 2)
        ]
        self.direction = (1, 0)  # Initial direction: right (x+1, y unchanged)
        self.food = self._generate_food()  # Spawn initial food
        self.score = 0
        self.done = False
        return self._get_state()  # Return initial game state

    def _generate_food(self):
        # Spawn food at a position not occupied by the snake
        while True:
            food_x = np.random.randint(0, self.state_size)
            food_y = np.random.randint(0, self.state_size)
            food_pos = (food_x, food_y)
            if food_pos not in self.snake:
                return food_pos

    def _get_state(self):
        # Construct game state (state_size x state_size grid):
        # 0 = empty, 1 = snake body, 2 = food, 3 = boundary (implicit via collision check)
        state = np.zeros((self.state_size, self.state_size), dtype=np.float32)
        # Mark snake body
        for (x, y) in self.snake:
            if 0 <= x < self.state_size and 0 <= y < self.state_size:
                state[y, x] = 1.0  # Note: numpy uses (row, column) = (y, x)
        # Mark food position
        food_x, food_y = self.food
        state[food_y, food_x] = 2.0
        return state

    def step(self, action):
        # Action mapping: 0=up, 1=right, 2=down, 3=left (convert to direction vector)
        action_directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.direction = action_directions[action]

        # Calculate new head position
        head_x, head_y = self.snake[0]
        new_head_x = head_x + self.direction[1]  # X direction (left/right)
        new_head_y = head_y + self.direction[0]  # Y direction (up/down)
        new_head = (new_head_x, new_head_y)

        # Collision detection (boundary or self)
        if (new_head_x < 0 or new_head_x >= self.state_size or
            new_head_y < 0 or new_head_y >= self.state_size or
            new_head in self.snake):
            self.done = True
            reward = -10.0  # Penalty for collision
        else:
            # Add new head to snake
            self.snake.insert(0, new_head)
            # Check if food is eaten
            if new_head == self.food:
                self.score += 1
                reward = 10.0  # Reward for eating food
                self.food = self._generate_food()  # Spawn new food
            else:
                self.snake.pop()  # Remove tail if no food is eaten
                reward = 0.1  # Small reward for surviving

        # Render game (only if visualization is enabled)
        if self.visualize:
            self._render()

        return self._get_state(), reward, self.done

    def _render(self):
        # Draw game screen
        self.display.fill(BLACK)
        # Draw snake
        for (x, y) in self.snake:
            pygame.draw.rect(
                self.display, GREEN,
                (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE - 1, BLOCK_SIZE - 1)  # -1 to avoid overlapping
            )
        # Draw food
        food_x, food_y = self.food
        pygame.draw.rect(
            self.display, RED,
            (food_x * BLOCK_SIZE, food_y * BLOCK_SIZE, BLOCK_SIZE - 1, BLOCK_SIZE - 1)
        )
        # Update display
        pygame.display.update()
        self.clock.tick(SPEED)

    def close(self):
        # Close Pygame window
        if self.visualize:
            pygame.quit()

# Test the game environment (run when this file is executed directly)
if __name__ == "__main__":
    game = SnakeGame(state_size=20, visualize=True)
    while not game.done:
        # Manual control (W=up, D=right, S=down, A=left)
        action = 1  # Default direction: right
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = 0
                elif event.key == pygame.K_d:
                    action = 1
                elif event.key == pygame.K_s:
                    action = 2
                elif event.key == pygame.K_a:
                    action = 3
        state, reward, done = game.step(action)
    print(f"Game Over! Final Score: {game.score}")
    game.close()