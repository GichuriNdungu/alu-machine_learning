import pygame
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)
        # Set start temp
        self.state = 38 + random.randint(-3, 3)
        # Set shower length
        self.shower_length = 60
        # Initialize pygame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Shower Environment")
        self.font = pygame.font.SysFont(None, 36)
        
    def step(self, action):
        # Apply action
        self.state += action - 1 
        # Reduce shower length by 1 second
        self.shower_length -= 1 
        
        # Calculate reward
        if 37 <= self.state <= 39: 
            reward = 1 
        else: 
            reward = -1 
        
        done = self.shower_length <= 0
        
        # Return step information
        return np.array([self.state]), reward, done, {}
    
    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.screen.fill((255, 255, 255))
        temp_bar_width = int(self.state * 4)  # Scale temperature to screen width
        pygame.draw.rect(self.screen, (0, 0, 255), (50, 150, temp_bar_width, 50))
        pygame.draw.line(self.screen, (0, 255, 0), (37 * 4 + 50, 150), (37 * 4 + 50, 200), 2)
        pygame.draw.line(self.screen, (0, 255, 0), (39 * 4 + 50, 150), (39 * 4 + 50, 200), 2)

        temp_text = self.font.render(f'Temperature: {self.state}°C', True, (0, 0, 0))
        time_text = self.font.render(f'Time remaining: {self.shower_length}s', True, (0, 0, 0))
        self.screen.blit(temp_text, (50, 50))
        self.screen.blit(time_text, (50, 100))

        pygame.display.flip()

    def reset(self):
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60 
        return np.array([self.state])

    def close(self):
        pygame.quit()

# Example usage:
env = ShowerEnv()
env.reset()
for _ in range(60):
    action = env.action_space.sample()
    env.step(action)
    env.render()
    pygame.time.wait(100)  # Slow down the rendering for visualization purposes
env.close()
