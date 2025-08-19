import pygame
import random
import numpy as np

pygame.init()

GRID_SIZE = 10
CELL_SIZE = 40

GRID_HEIGHT = GRID_SIZE*CELL_SIZE
GRID_WIDTH = GRID_HEIGHT

FPS = 10

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class SnakeGame:
    def __init__(self, render=True):
        self.render = render
        if self.render:
            self.screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
            pygame.display.set_caption('Snake Dreamer')
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE//2, GRID_SIZE//2)]
        self.score = 0
        self.direction = RIGHT
        self.food = self._place_food()
        
        return self._state()

    def _place_food(self):
        while True:
            food = (random.randint(0,GRID_SIZE-1), random.randint(0,GRID_SIZE-1))

            if food not in self.snake:
                return food
            
    def _state(self):
        """
        state: [danger_up, danger_right, danger_down, danger_left, 
                 food_up, food_right, food_down, food_left,
                 direction_up, direction_right, direction_down, direction_left,
                 snake_length]
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        state = [0]*13

        # dangerous or not in a direction of snake head
        for i, (dx, dy) in enumerate([(-1,0),(0,1),(1,0),(0,-1)]):
            next_x = (head_x+dx) % GRID_SIZE
            next_y = (head_y+dy) % GRID_SIZE
            if (next_x, next_y) in self.snake[1:]:
                state[i] = 1

        # food is or is not in a direction of snake head
        state[4] = 1 if food_x < head_x else 0
        state[5] = 1 if food_y > head_y else 0
        state[6] = 1 if food_x > head_x else 0
        state[7] = 1 if food_y < head_y else 0

        # current direction
        state[8 + self.direction] = 1

        # curent length
        state[12] = len(self.snake) / GRID_SIZE
        
        return np.array(state, dtype=np.float32)
            
    def step(self, action):
        # 3 actions prevent agent reversing directions
        if action == 0:
            pass
        elif action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        head_x, head_y = self.snake[0]
        if self.direction == UP:
            head_x -= 1
        elif self.direction == RIGHT:
            head_y += 1
        elif self.direction == DOWN:
            head_x += 1
        elif self.direction == LEFT:
            head_y -= 1

        head_x %= GRID_SIZE
        head_y %= GRID_SIZE
        new_head = (head_x, head_y)

        if new_head in self.snake[1:]:
            return self._get_state(), -1 + 0.1 * len(self.snake), True
        
        self.snake.insert(0, new_head)

        reward = -0.01

        game_over = False

        if new_head == self.food:
            self.score += 1
            reward += 1 + 0.1 * len(self.snake) # better reward the longer the snake
            self.food = self._place_food()
        else:
            self.snake.pop()
        
        return self._state(), reward, game_over
    
    def _render(self):
        self.screen.fill(BLACK)
        for x, y in self.snake:
            pygame.draw.rect(self.screen, GREEN, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        fx, fy = self.food
        pygame.draw.rect(self.screen, RED, (fy * CELL_SIZE, fx * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()
        self.clock.tick(FPS)