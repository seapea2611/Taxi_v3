import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(6)
        # Example for using image as input:
        self.observation_space = spaces.Discrete(25)  # Custom map size

        # Initialize the state
        self.state = self.reset()

    def reset(self):
        # Initialize the state to a random location
        self.state = self.observation_space.sample()
        return self.state

    def step(self, action):
        # Implement how the environment responds to an action
        reward = 0
        done = False
        info = {}

        # Implement the dynamics of your environment here
        self.state = (self.state + action) % self.observation_space.n

        return self.state, reward, done, info

    def render(self, mode='human'):
        # Implement rendering of the environment
        print(f'State: {self.state}')

    def close(self):
        pass

# Example of using the custom environment
if __name__ == "__main__":
    env = CustomEnv()
    for _ in range(10):
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)
            env.render()
