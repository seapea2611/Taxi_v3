import numpy as np
import random
import time
from custom_taxi_env import CustomTaxiEnv

env = CustomTaxiEnv()
env.render()

state_space = env.observation_space.n
print("There are ", state_space, " possible states")
action_space = env.action_space.n
print("There are ", action_space, " possible actions")

# Create our Q table with state_size rows and action_size columns
Q = np.zeros((state_space, action_space))
print(Q)
print(Q.shape)

total_episodes = 25000        # Total number of training episodes
total_test_episodes = 10      # Total number of test episodes
max_steps = 200               # Max steps per episode

learning_rate = 0.01          # Learning rate
gamma = 0.99                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.001           # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration prob

def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(Q[state])
    else:
        action = env.action_space.sample()
    return action

# Training
for episode in range(total_episodes):
    state = env.reset()
    done = False
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    for step in range(max_steps):
        action = epsilon_greedy_policy(Q, state, epsilon)
        new_state, reward, done, info = env.step(action)
        Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * np.max(Q[new_state]) - Q[state][action])
        state = new_state
        if done:
            break

# Testing
rewards = []
for episode in range(total_test_episodes):
    state = env.reset()
    done = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)
    for step in range(max_steps):
        env.render('pygame')
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        if done:
            rewards.append(total_rewards)
            break
        state = new_state
        time.sleep(0.5)
env.close()
print("Score over time: " + str(sum(rewards) / total_test_episodes))
