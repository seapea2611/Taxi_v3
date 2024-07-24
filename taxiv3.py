import numpy as np
import random
import time
from custom_map import CustomEnv

# Sử dụng môi trường tùy chỉnh
env = CustomEnv()
env.render()

state_space = env.observation_space.n
print("There are ", state_space, " possible states")
action_space = env.action_space.n
print("There are ", action_space, " possible actions")

# Tạo bảng Q với state_size hàng và action_size cột
Q = np.zeros((state_space, action_space))
print(Q)
print(Q.shape)

total_episodes = 25000        # Tổng số tập huấn luyện
total_test_episodes = 10      # Tổng số tập kiểm tra
max_steps = 200               # Số bước tối đa mỗi tập

learning_rate = 0.01          # Tốc độ học
gamma = 0.99                  # Tỷ lệ chiết khấu

# Tham số khám phá
epsilon = 1.0                 # Tỷ lệ khám phá
max_epsilon = 1.0             # Xác suất khám phá ban đầu
min_epsilon = 0.001           # Xác suất khám phá tối thiểu
decay_rate = 0.01             # Tỷ lệ giảm dần của xác suất khám phá

def epsilon_greedy_policy(Q, state, epsilon):
    # Nếu số ngẫu nhiên > epsilon --> khai thác
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(Q[state])
    # Ngược lại --> khám phá
    else:
        action = env.action_space.sample()

    return action

# Huấn luyện
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

# Kiểm tra
rewards = []
for episode in range(total_test_episodes):
    state = env.reset()
    done = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)
    for step in range(max_steps):
        env.render()
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        if done:
            rewards.append(total_rewards)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards) / total_test_episodes))
