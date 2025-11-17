import pygame
import random
import numpy as np

# Game configuration
GRID_SIZE = 20  # 20x20 grid
CELL_SIZE = 30  # Pixels per cell
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 10  # Game speed

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)  # Snake body
RED = (255, 0, 0)    # Food
BLUE = (0, 0, 255)   # Snake head

class SnakeGame:
    def __init__(self, state_size, visualize=True):
        self.state_size = state_size
        self.visualize = visualize
        
        # Initialize Pygame if visualization is enabled
        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake Game - DQN Agent")
            self.clock = pygame.time.Clock()
        
        self.reset()  # Initialize game state

    def reset(self):
        """Reset game to initial state"""
        # Snake initial position and direction
        self.snake_head = (GRID_SIZE // 2, GRID_SIZE // 2)
        self.snake_body = [self.snake_head, (self.snake_head[0]-1, self.snake_head[1])]
        self.direction = (1, 0)  # Start moving right
        
        # Spawn initial food
        self.food_pos = self._spawn_food()
        
        # Reset game variables
        self.collision = False
        self.food_eaten = False
        self.step_count = 0
        
        return self.get_state()

    def _spawn_food(self):
        """Spawn food in an empty cell (not on snake body)"""
        while True:
            food_pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if food_pos not in self.snake_body:
                return food_pos

    def get_state(self):
        """Convert game state to feature vector"""
        head_x, head_y = self.snake_head
        food_x, food_y = self.food_pos
        
        # Direction one-hot encoding (4 features)
        dir_up = 1 if self.direction == (0, -1) else 0
        dir_down = 1 if self.direction == (0, 1) else 0
        dir_left = 1 if self.direction == (-1, 0) else 0
        dir_right = 1 if self.direction == (1, 0) else 0
        
        # Danger detection (4 features)
        danger_up = 1 if (head_y - 1 < 0) or ((head_x, head_y - 1) in self.snake_body) else 0
        danger_down = 1 if (head_y + 1 >= GRID_SIZE) or ((head_x, head_y + 1) in self.snake_body) else 0
        danger_left = 1 if (head_x - 1 < 0) or ((head_x - 1, head_y) in self.snake_body) else 0
        danger_right = 1 if (head_x + 1 >= GRID_SIZE) or ((head_x + 1, head_y) in self.snake_body) else 0
        
        # Food relative position (4 features)
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0
        
        # Additional distance features (8 features)
        dist_food_x = (food_x - head_x) / GRID_SIZE
        dist_food_y = (food_y - head_y) / GRID_SIZE
        dist_wall_left = head_x / GRID_SIZE
        dist_wall_right = (GRID_SIZE - 1 - head_x) / GRID_SIZE
        dist_wall_up = head_y / GRID_SIZE
        dist_wall_down = (GRID_SIZE - 1 - head_y) / GRID_SIZE
        body_length = len(self.snake_body) / GRID_SIZE
        long_body = 1 if len(self.snake_body) > 3 else 0
        
        # Combine all features (total 20)
        state = [
            dir_up, dir_down, dir_left, dir_right,
            danger_up, danger_down, danger_left, danger_right,
            food_up, food_down, food_left, food_right,
            dist_food_x, dist_food_y, dist_wall_left, dist_wall_right,
            dist_wall_up, dist_wall_down, body_length, long_body
        ]
        
        assert len(state) == self.state_size, f"State size mismatch: {len(state)} vs {self.state_size}"
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Execute action and return (next_state, reward, done)"""
        self.step_count += 1
        self.food_eaten = False
        self.collision = False
        
        # Handle Pygame events
        if self.visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        # Map action to direction (0=up, 1=down, 2=left, 3=right)
        direction_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_direction = direction_map[action]
        
        # Prevent 180-degree turns (e.g., right → left)
        if (new_direction[0] == -self.direction[0]) and (new_direction[1] == -self.direction[1]):
            new_direction = self.direction
        self.direction = new_direction

        # Calculate new head position
        new_head_x = self.snake_head[0] + self.direction[0]
        new_head_y = self.snake_head[1] + self.direction[1]
        self.new_head = (new_head_x, new_head_y)

        # Check for collision
        if (new_head_x < 0 or new_head_x >= GRID_SIZE or
            new_head_y < 0 or new_head_y >= GRID_SIZE or
            self.new_head in self.snake_body):
            self.collision = True

        # Update snake body
        self.snake_body.insert(0, self.new_head)
        self.snake_head = self.new_head

        # Check food consumption
        if self.snake_head == self.food_pos:
            self.food_eaten = True
            self.food_pos = self._spawn_food()
        else:
            self.snake_body.pop()  # Remove tail if no food eaten

        # Calculate reward
        reward = self._calculate_reward()

        # Draw if visualization is enabled
        if self.visualize:
            self._draw()

        # Episode ends on collision or step limit
        done = self.collision or self.step_count >= 300
        return self.get_state(), reward, done

    def _calculate_reward(self):
        """Optimized reward function to guide agent behavior"""
        reward = 0.0
        
        # Large reward for eating food
        if self.food_eaten:
            reward += 10.0
        
        # Large penalty for collision
        if self.collision:
            reward -= 20.0
        
        # Reward for moving closer to food
        prev_head = self.snake_body[1] if len(self.snake_body) > 1 else self.snake_head
        prev_distance = abs(prev_head[0] - self.food_pos[0]) + abs(prev_head[1] - self.food_pos[1])
        new_distance = abs(self.snake_head[0] - self.food_pos[0]) + abs(self.snake_head[1] - self.food_pos[1])
        if new_distance < prev_distance:
            reward += 1.0
        else:
            reward -= 0.5
        
        # Penalty for getting too close to walls
        wall_distance_x = min(self.snake_head[0], GRID_SIZE - 1 - self.snake_head[0])
        wall_distance_y = min(self.snake_head[1], GRID_SIZE - 1 - self.snake_head[1])
        if wall_distance_x < 3 or wall_distance_y < 3:
            reward -= 0.8
        
        # Penalty for wandering without eating
        if self.step_count % 50 == 0 and not self.food_eaten:
            reward -= 1.5
        
        return reward

    def _draw(self):
        """Render game elements"""
        self.screen.fill(BLACK)
        
        # Draw snake
        for i, segment in enumerate(self.snake_body):
            x = segment[0] * CELL_SIZE
            y = segment[1] * CELL_SIZE
            color = BLUE if i == 0 else GREEN  # Head is blue
            pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE - 1, CELL_SIZE - 1))
        
        # Draw food
        food_x = self.food_pos[0] * CELL_SIZE
        food_y = self.food_pos[1] * CELL_SIZE
        pygame.draw.rect(self.screen, RED, (food_x, food_y, CELL_SIZE - 1, CELL_SIZE - 1))
        
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        """Clean up Pygame resources"""
        if self.visualize:
            pygame.quit()

if __name__ == "__main__":
    # Test the game environment
    game = SnakeGame(state_size=20, visualize=True)
    state = game.reset()
    done = False
    while not done:
        action = random.choice(range(4))  # Random action for testing
        next_state, reward, done = game.step(action)
        print(f"Step: {game.step_count}, Reward: {reward:.2f}, Done: {done}")
    game.close()