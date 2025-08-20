import tensorflow as tf
from model.actor import Actor
from model.critic import Critic
import numpy as np
from collections import deque
import random

class PPOAgent:
    def __init__(self, state_size=13, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        self.actor = Actor(self.state_size, self.action_size)
        self.critic = Critic(self.state_size)

        self.memory = deque(maxlen=2000)

        self.eps = 0.2
        self.gamma = 0.99
        
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        probs = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=probs)
        
        return probs, action
    
    def store_exp(self, state, action, reward, next_state, done, prob):
        self.memory.append(state, action, reward, next_state, done, prob)

    def train(self):
        batch = random.sample(self.memory, self.batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        old_probs = np.array([t[5] for t in batch])

        values = self.critic.predict(states)
        next_vals = self.critic.predict(next_states)

        advantages = rewards + self.gamma * next_vals * (1-dones) - values # TD error formula + bellman eq

        with tf.GradientTape() as tape:
            probs = self.actor.model(states, training=True)

            old_probs_ac = tf.reduce_sum(old_probs * tf.one_hot(actions, self.action_size), axis=1)
            new_probs_ac = tf.reduce_sum(probs * tf.one_hot(actions, self.action_size), axis=1)

            ratios = new_probs_ac / (old_probs_ac + 1e-10)
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon)

            actor_loss = -tf.reduce_mean(
                tf.minimum(ratios * advantages, clipped_ratios * advantages)
            )

        actor_grads = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.model.trainable_variables))

        with tf.GradientTape() as tape:
            values = self.critic.model(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(values - (rewards + self.gamma * next_vals * (1 - dones))))

        critic_grads = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.model.trainable_variables))
