import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

# Initialize the environment
env = gym.make('CartPole-v1')
np.random.seed(123)
if hasattr(env, 'seed'):
    env.seed(123)
nb_actions = env.action_space.n  # Total number of actions for CartPole-v1

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Configure and compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# Test the agent
dqn.test(env, nb_episodes=5, visualize=True)