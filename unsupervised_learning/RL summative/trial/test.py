from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam
from model import build_model
from env import ShowerEnv

# Instantiate the environment
env = ShowerEnv()

# Define the model
states = env.observation_space.shape
actions = env.action_space.n
model = build_model(states, actions)

# Build the DQN agent
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10,
                   target_model_update=1e-2)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Load the saved weights
dqn.load_weights('saved_weights/dqn_weights.h5f')

# Test the agent
episodes = 5
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = dqn.forward(state)
        state, reward, done, info = env.step(action)
        env.render()

print('Testing completed')
