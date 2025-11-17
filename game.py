import numpy as np
import pygame
import random

class SnakeGame:
    def __init__(self, state_size=20, visualize=False):
        self.state_size = state_size  # Fixed 20x20 grid
        self.grid_size = 20  # Pixels per cell
        self.width = state_size  # Valid indices: 0-19
        self.height = state_size
        
        self.visualize = visualize
        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((
                self.width * self.grid_size, 
                self.height * self.grid_size
            ))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
        
        self.reset()

    def _generate_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                return (x, y)

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (0, 1)  # Right
        self.food = self._generate_food()
        self.score = 0
        return self._get_state()

    def _get_state(self):
        # Strictly return (1, 20, 20) shape (1 channel)
        state = np.zeros((1, self.state_size, self.state_size), dtype=np.float32)
        # Mark snake (1.0)
        for (x, y) in self.snake:
            if 0 <= x < self.width and 0 <= y < self.height:
                state[0, x, y] = 1.0  # Channel 0, x row, y column
        # Mark food (2.0)
        fx, fy = self.food
        if 0 <= fx < self.width and 0 <= fy < self.height:
            state[0, fx, fy] = 2.0
        return state

    def step(self, action):
        # Action mapping: 0=up, 1=right, 2=down, 3=left
        if action == 0 and self.direction != (1, 0):  # Up
            self.direction = (-1, 0)
        elif action == 1 and self.direction != (0, -1):  # Right
            self.direction = (0, 1)
        elif action == 2 and self.direction != (-1, 0):  # Down
            self.direction = (1, 0)
        elif action == 3 and self.direction != (0, 1):  # Left
            self.direction = (0, -1)

        # Move head
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        self.snake.insert(0, new_head)

        # Reward & done logic
        reward = 0.1  # Survival reward
        done = False

        # Wall collision (0 <= x < 20, 0 <= y < 20)
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height):
            done = True
            reward = -10

        # Self collision
        if new_head in self.snake[1:]:
            done = True
            reward = -10

        # Eat food
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._generate_food()
        else:
            self.snake.pop()  # Remove tail

        # Visualize
        if self.visualize:
            self._render()

        return self._get_state(), reward, done

    def _render(self):
        self.screen.fill((0, 0, 0))  # Black background
        # Draw snake
        for (x, y) in self.snake:
            pygame.draw.rect(
                self.screen, (0, 255, 0),  # Green
                (x*self.grid_size, y*self.grid_size, 
                 self.grid_size-1, self.grid_size-1)
            )
        # Draw food
        fx, fy = self.food
        pygame.draw.rect(
            self.screen, (255, 0, 0),  # Red
            (fx*self.grid_size, fy*self.grid_size, 
             self.grid_size-1, self.grid_size-1)
        )
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.visualize:
            pygame.quit()