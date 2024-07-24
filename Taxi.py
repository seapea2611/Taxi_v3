# import numpy as np
# import gym
# import random
# import time
# import pygame
#
# # Khởi tạo môi trường taxi-v3 từ OpenAI Gym
# env = gym.make("Taxi-v3")
# env.render()
#
# state_space = env.observation_space.n
# action_space = env.action_space.n
#
# # Tạo bảng Q với kích thước (state_space x action_space) (500x6)
# Q = np.zeros((state_space, action_space))
#
# total_episodes = 25000  # Tổng số tập huấn luyện
# total_test_episodes = 10  # Tổng số tập kiểm tra
# max_steps = 200  # Số bước tối đa mỗi tập
#
# learning_rate = 0.01  # Tốc độ học
# gamma = 0.99  # Hệ số chiết khấu
#
# # Tham số khám phá
# epsilon = 1.0  # Tỷ lệ khám phá
# max_epsilon = 1.0  # Xác suất khám phá ban đầu
# min_epsilon = 0.001  # Xác suất khám phá tối thiểu
# decay_rate = 0.01  # Tỷ lệ giảm khám phá
#
#
# def epsilon_greedy_policy(Q, state, epsilon):
#     # nếu số ngẫu nhiên > epsilon --> khai thác
#     if random.uniform(0, 1) > epsilon:
#         action = np.argmax(Q[state])
#     # nếu không --> khám phá
#     else:
#         action = env.action_space.sample()
#     return action
#
#
# # Huấn luyện
# for episode in range(total_episodes):
#     state = env.reset()
#     step = 0
#     done = False
#     epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
#     for step in range(max_steps):
#         action = epsilon_greedy_policy(Q, state, epsilon)
#         new_state, reward, done, info = env.step(action)
#         Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * np.max(Q[new_state]) - Q[state][action])
#         if done:
#             break
#         state = new_state
#
# print("Training finished.")
#
# # Khởi tạo Pygame
# pygame.init()
#
# WINDOW_SIZE = 500
# GRID_SIZE = WINDOW_SIZE // 5
#
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# BLUE = (0, 0, 255)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)
# YELLOW = (255, 255, 0)
#
# # Thêm màu nền
# BACKGROUND_COLOR = (0, 128, 0)  # Màu xanh lá cây
#
# FONT = pygame.font.SysFont('Arial', 25)
#
# screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
# pygame.display.set_caption('Taxi-v3')
#
# # Tải icon taxi
# taxi_icon = pygame.image.load('taxi_icon.png')
#
# # Các bức tường trong môi trường Taxi-v3
# walls = [
#     ((1, 0), (2, 0)),
#     ((1, 1), (2, 1)),
#     ((0, 3), (1, 3)),
#     ((0, 4), (1, 4)),
#     ((2, 3), (3, 3)),
#     ((2, 4), (3, 4))
# ]
#
# def draw_grid(state, steps, reward, episode):
#     screen.fill(BACKGROUND_COLOR)
#
#     # Vẽ các bức tường
#     for wall in walls:
#         (x1, y1), (x2, y2) = wall
#         if x1 == x2:  # Vertical wall
#             start_pos = (x1 * GRID_SIZE, min(y1, y2) * GRID_SIZE + GRID_SIZE)
#             end_pos = (x1 * GRID_SIZE + GRID_SIZE, min(y1, y2) * GRID_SIZE + GRID_SIZE )
#         else:  # Horizontal wall
#             start_pos = (min(x1, x2) * GRID_SIZE + GRID_SIZE , y1 * GRID_SIZE)
#             end_pos = (min(x1, x2) * GRID_SIZE + GRID_SIZE, y1 * GRID_SIZE + GRID_SIZE)
#         pygame.draw.line(screen, BLACK, start_pos, end_pos, 5)
#
#     taxi_row, taxi_col, pass_loc, dest_loc = env.decode(state)
#     taxi_rect = pygame.Rect(taxi_col * GRID_SIZE, taxi_row * GRID_SIZE, GRID_SIZE, GRID_SIZE)
#     screen.blit(pygame.transform.scale(taxi_icon, (GRID_SIZE, GRID_SIZE)), taxi_rect)
#
#     loc_colors = [RED, GREEN, YELLOW, BLACK]
#     for i, loc in enumerate(env.locs):
#         loc_rect = pygame.Rect(loc[1] * GRID_SIZE, loc[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
#         pygame.draw.rect(screen, loc_colors[i], loc_rect, 2 if i != dest_loc else 5)
#
#     if pass_loc < 4:
#         pass_rect = pygame.Rect(env.locs[pass_loc][1] * GRID_SIZE, env.locs[pass_loc][0] * GRID_SIZE, GRID_SIZE,
#                                 GRID_SIZE)
#         pygame.draw.circle(screen, loc_colors[pass_loc], pass_rect.center, GRID_SIZE // 4)
#     else:
#         pygame.draw.circle(screen, loc_colors[dest_loc], taxi_rect.center, GRID_SIZE // 4)
#
#     reward_text = FONT.render(f'Episode: {episode} Step: {steps} Reward: {reward}', True, WHITE)
#     screen.blit(reward_text, (10, 10))
#
#     pygame.display.flip()
#
#
# # Main game loop
# running = True
# episode = 0
#
# while running and episode < total_test_episodes:
#     state = env.reset()
#     steps = 0
#     total_rewards = 0
#     done = False
#     while not done:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#                 done = True
#
#         if isinstance(state, (int, np.integer)):  # Ensure the state is an integer
#             action = np.argmax(Q[state])  # Use the learned policy
#
#             next_state, reward, done, _ = env.step(action)
#             state = next_state
#             steps += 1
#             total_rewards += reward
#
#             draw_grid(state, steps, total_rewards, episode)
#             time.sleep(0.36)  # Delay to make the game viewable
#
#     episode += 1
#
# pygame.quit()
#
# print(Q)


import numpy as np
import gym
import random
import time
import pygame

# Khởi tạo môi trường taxi-v3 từ OpenAI Gym
env = gym.make("Taxi-v3")

state_space = env.observation_space.n
action_space = env.action_space.n

# Tạo bảng Q với kích thước (state_space x action_space) (500x6)
Q = np.zeros((state_space, action_space))

total_episodes = 25000  # Tổng số tập huấn luyện
total_test_episodes = 10  # Tổng số tập kiểm tra
max_steps = 200  # Số bước tối đa mỗi tập

learning_rate = 0.01  # Tốc độ học
gamma = 0.99  # Hệ số chiết khấu

# Tham số khám phá
epsilon = 1.0  # Tỷ lệ khám phá
max_epsilon = 1.0  # Xác suất khám phá ban đầu
min_epsilon = 0.001  # Xác suất khám phá tối thiểu
decay_rate = 0.01  # Tỷ lệ giảm khám phá

def epsilon_greedy_policy(Q, state, epsilon):
    # nếu số ngẫu nhiên > epsilon --> khai thác
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(Q[state])
    # nếu không --> khám phá
    else:
        action = env.action_space.sample()
    return action

# Khởi tạo Pygame
pygame.init()

WINDOW_SIZE = 500
GRID_SIZE = WINDOW_SIZE // 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Thêm màu nền
BACKGROUND_COLOR = (0, 128, 0)  # Màu xanh lá cây

FONT = pygame.font.SysFont('Arial', 25)

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('Taxi-v3')

# Tải icon taxi
taxi_icon = pygame.image.load('taxi_icon.png')

# Các bức tường trong môi trường Taxi-v3
walls = [
    ((1, 0), (2, 0)),
    ((1, 1), (2, 1)),
    ((0, 3), (1, 3)),
    ((0, 4), (1, 4)),
    ((2, 3), (3, 3)),
    ((2, 4), (3, 4))
]

def draw_grid(state, steps, reward, episode):
    screen.fill(BACKGROUND_COLOR)

    # Vẽ các bức tường
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        if x1 == x2:  # Vertical wall
            start_pos = (x1 * GRID_SIZE, min(y1, y2) * GRID_SIZE + GRID_SIZE)
            end_pos = (x1 * GRID_SIZE + GRID_SIZE, min(y1, y2) * GRID_SIZE + GRID_SIZE )
        else:  # Horizontal wall
            start_pos = (min(x1, x2) * GRID_SIZE + GRID_SIZE , y1 * GRID_SIZE)
            end_pos = (min(x1, x2) * GRID_SIZE + GRID_SIZE, y1 * GRID_SIZE + GRID_SIZE)
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 5)

    taxi_row, taxi_col, pass_loc, dest_loc = env.decode(state)
    taxi_rect = pygame.Rect(taxi_col * GRID_SIZE, taxi_row * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    screen.blit(pygame.transform.scale(taxi_icon, (GRID_SIZE, GRID_SIZE)), taxi_rect)

    loc_colors = [RED, GREEN, YELLOW, BLACK]
    for i, loc in enumerate(env.locs):
        loc_rect = pygame.Rect(loc[1] * GRID_SIZE, loc[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, loc_colors[i], loc_rect, 2 if i != dest_loc else 5)

    if pass_loc < 4:
        pass_rect = pygame.Rect(env.locs[pass_loc][1] * GRID_SIZE, env.locs[pass_loc][0] * GRID_SIZE, GRID_SIZE,
                                GRID_SIZE)
        pygame.draw.circle(screen, loc_colors[pass_loc], pass_rect.center, GRID_SIZE // 4)
    else:
        pygame.draw.circle(screen, loc_colors[dest_loc], taxi_rect.center, GRID_SIZE // 4)

    reward_text = FONT.render(f'Episode: {episode} Step: {steps} Reward: {reward}', True, WHITE)
    screen.blit(reward_text, (10, 10))

    pygame.display.flip()

# Huấn luyện với Pygame
for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                quit()

        action = epsilon_greedy_policy(Q, state, epsilon)
        new_state, reward, done, info = env.step(action)
        Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

        draw_grid(state, step, reward, episode)
        state = new_state
        step += 1

        time.sleep(0.01)

print("Training finished.")
pygame.quit()

# Kiểm tra không sử dụng Pygame
rewards = []
for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    total_rewards = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)
    while not done:
        env.render()
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        state = new_state
        total_rewards += reward
        step += 1

        if done:
            rewards.append(total_rewards)
            break

print("Score over time: " + str(sum(rewards) / total_test_episodes))
env.close()
