from snake_game import SnakeGame
from dqn_agent import DQNAgent
import numpy as np
import time
import pygame

FPS = 10

def main(render):
    game = SnakeGame(render=render)
    agent = DQNAgent()

    agent.load('dqn_model.h5')

    episodes = 1000

    scores = []

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = game.step(action)

            agent.store_exp(state, action, reward, next_state, done)
            agent.train()

            if render:
                game._render()

            state = next_state

            total_reward += reward
            
        scores.append(game.score)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}\nAverage Score: {np.mean(scores[-100:])}")

    agent.save('dqn_model.h5')

    state = game.reset()
    done = False

    while not done:
        action = agent.get_action(state)
        state, reward, done = game.step(action)
        time.sleep(1.0/FPS)

    pygame.quit()

if __name__ == '__main__':
    main(render=True)
