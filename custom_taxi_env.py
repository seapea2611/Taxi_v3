import gym
from gym import spaces
import numpy as np
import pygame
import time

class CustomTaxiEnv(gym.Env):
    metadata = {'render.modes': ['human', 'pygame']}

    def __init__(self):
        super(CustomTaxiEnv, self).__init__()

        self.grid_size = 10  # Tăng kích thước của bản đồ taxi
        self.state_space_size = self.grid_size * self.grid_size * 4 * 5 * 4
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(self.state_space_size)

        self.reset()

        self.screen = None
        self.cell_size = 50
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size

    def reset(self):
        # Tạo lại trạng thái ban đầu
        self.taxi_row = np.random.randint(self.grid_size)
        self.taxi_col = np.random.randint(self.grid_size)
        self.passenger_loc = np.random.randint(4)
        self.destination = np.random.randint(4)
        self.state = self.encode(self.taxi_row, self.taxi_col, self.passenger_loc, self.destination)
        return self.state

    def step(self, action):
        # Thực hiện hành động và tính toán trạng thái mới, phần thưởng, và liệu có kết thúc không
        self.state = self.decode(self.state)
        reward = -1
        done = False

        if action == 0:  # South
            self.taxi_row = min(self.taxi_row + 1, self.grid_size - 1)
        elif action == 1:  # North
            self.taxi_row = max(self.taxi_row - 1, 0)
        elif action == 2:  # East
            self.taxi_col = min(self.taxi_col + 1, self.grid_size - 1)
        elif action == 3:  # West
            self.taxi_col = max(self.taxi_col - 1, 0)
        elif action == 4:  # Pickup
            if self.passenger_loc == 4:
                reward = -10
            elif (self.taxi_row, self.taxi_col) == self.loc2coord(self.passenger_loc):
                self.passenger_loc = 4
                reward = 20
            else:
                reward = -10
        elif action == 5:  # Dropoff
            if self.passenger_loc != 4:
                reward = -10
            elif (self.taxi_row, self.taxi_col) == self.loc2coord(self.destination):
                self.passenger_loc = self.destination
                reward = 20
                done = True
            else:
                reward = -10

        self.state = self.encode(self.taxi_row, self.taxi_col, self.passenger_loc, self.destination)

        return self.state, reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            print(f'State: {self.state}, Taxi Position: ({self.taxi_row}, {self.taxi_col}), Passenger Location: {self.passenger_loc}, Destination: {self.destination}')
        elif mode == 'pygame':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))

            self.screen.fill((255, 255, 255))

            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

            taxi_rect = pygame.Rect(self.taxi_col * self.cell_size, self.taxi_row * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 255, 0), taxi_rect)

            if self.passenger_loc != 4:
                passenger_row, passenger_col = self.loc2coord(self.passenger_loc)
                passenger_rect = pygame.Rect(passenger_col * self.cell_size, passenger_row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 255, 0), passenger_rect)

            destination_row, destination_col = self.loc2coord(self.destination)
            destination_rect = pygame.Rect(destination_col * self.cell_size, destination_row * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), destination_rect)

            pygame.display.flip()

    def encode(self, taxi_row, taxi_col, passenger_loc, destination):
        # Mã hóa trạng thái
        return (taxi_row * self.grid_size + taxi_col) * 20 + passenger_loc * 5 + destination

    def decode(self, state):
        # Giải mã trạng thái
        out = [0, 0, 0, 0]
        out[3] = state % 5
        state = state // 5
        out[2] = state % 4
        state = state // 4
        out[1] = state % self.grid_size
        out[0] = state // self.grid_size
        return out

    def loc2coord(self, loc):
        # Chuyển đổi vị trí thành tọa độ
        if loc == 0:
            return (0, 0)
        elif loc == 1:
            return (0, self.grid_size - 1)
        elif loc == 2:
            return (self.grid_size - 1, 0)
        elif loc == 3:
            return (self.grid_size - 1, self.grid_size - 1)
        else:
            return None

    def close(self):
        if self.screen is not None:
            pygame.quit()

# Example of using the custom environment
if __name__ == "__main__":
    env = CustomTaxiEnv()
    for _ in range(10):
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)
            env.render('pygame')
            time.sleep(0.1)
    env.close()
