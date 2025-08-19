import pygame
import random

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

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE//2, GRID_SIZE//2)]
        self.direction = RIGHT
        self.food = self._place_food()

    def _place_food(self):
        while True:
            food = (random.randint(0,GRID_SIZE-1), random.randint(0,GRID_SIZE-1))

            if food not in self.snake:
                return food