import tensorflow as tf
import numpy as np
from model.actor import Actor
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size=13, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = 64

        self.eps = 1.0
        self.gamma = 0.99

        self.learning_rate = 0.001

        self.model = Actor(self.state_size, self.action_size).model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                           loss='mse')
        
        self.memory = deque(maxlen=2000)

        self.eps_min = 0.05
        self.eps_decay = 0.995

    def get_action(self, state):
        if np.random.rand() < self.eps:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def store_exp(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        q_next = self.model.predict(next_states, verbose=0)
        q_targets = self.model.predict(states, verbose=0)

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(q_next[i])
            q_targets[i][actions[i]] = target

        self.model.fit(states, q_targets, epochs=1, verbose=0)

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        
    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
    


    