#!/usr/bin/env python3
# python script to develop a customized gym environment 

import gym
from gym import spaces
import numpy as np 

d = 3 
class ClimateCompensationEnv(gym.Env):
    def __init__(self):
        super(ClimateCompensationEnv, self).__init__()
        # define the action and observation space
        # Actions: invest in a project A, B, C
        # hypthetically 3 projects
        self.action_space = spaces.Box(low=0, high=1, shape=(3 * d,), dtype=np.float32)  # Flattened action space
        
        # observations will be current emmissions level, project impact etc 
        self.observation_space  = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)

        self.state = None 
        self.emmissions_target = 50 #example emmisions target 
        self.reset()
    def reset(self):
        '''
        resets the gym environment to its initial state
        ensures that the environment starts from a known state
        hence, agent has a consistent starting point
        '''
        self.state = np.array([100, 0])
        return self.state
    def step(self, action):
        '''
        defines how the environment responds to actions
        '''
        emmissions, total_reductions = self.state

        if action == 0: #invest in project A
            reduction = np.random.uniform(1, 5)
        elif action == 1:
            reduction = np.random.uniform(3, 7)
        else:
            reduction = np.random.uniform(2, 6)
        
        emmissions -= reduction
        total_reductions += reduction
        self.state = np.array([emmissions, total_reductions])

        reward = reduction
        done = emmissions <= self.emmissions_target

        return self.state, reward, done, {}
    
    def render(self, mode='human', close=False):
        print(f"Emissions: {self.state[0]}, Total Reductions: {self.state[1]}")
    def seed(self, seed=None):
        np.random.seed(seed)