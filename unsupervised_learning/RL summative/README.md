# Shower Environment Reinforcement Learning

This project implements a Deep Q-Network (DQN) agent to interact with a simulated shower environment. The goal is to maintain the shower temperature within a comfortable range using reinforcement learning.

## Project Structure

- [`model.py`](command:_github.copilot.openSymbolFromReferences?%5B%22model.py%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cuser%5C%5CDesktop%5C%5Ccodes%5C%5Calu-machine_learning%5C%5Cunsupervised_learning%5C%5CRL%20summative%5C%5Ctrial%5C%5Ctest.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%2520summative%2Ftrial%2Ftest.py%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%20summative%2Ftrial%2Ftest.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A4%2C%22character%22%3A5%7D%7D%5D%5D "Go to definition"): Contains the neural network model definition.
- [`env.py`](command:_github.copilot.openSymbolFromReferences?%5B%22env.py%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cuser%5C%5CDesktop%5C%5Ccodes%5C%5Calu-machine_learning%5C%5Cunsupervised_learning%5C%5CRL%20summative%5C%5Ctrial%5C%5Ctest.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%2520summative%2Ftrial%2Ftest.py%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%20summative%2Ftrial%2Ftest.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A5%2C%22character%22%3A5%7D%7D%5D%5D "Go to definition"): Defines the [`ShowerEnv`](command:_github.copilot.openSymbolFromReferences?%5B%22ShowerEnv%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cuser%5C%5CDesktop%5C%5Ccodes%5C%5Calu-machine_learning%5C%5Cunsupervised_learning%5C%5CRL%20summative%5C%5Ctrial%5C%5Ctest.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%2520summative%2Ftrial%2Ftest.py%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%20summative%2Ftrial%2Ftest.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A5%2C%22character%22%3A16%7D%7D%5D%5D "Go to definition") environment.
- [`agent.py`](command:_github.copilot.openSymbolFromReferences?%5B%22agent.py%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cuser%5C%5CDesktop%5C%5Ccodes%5C%5Calu-machine_learning%5C%5Cunsupervised_learning%5C%5CRL%20summative%5C%5Ctrial%5C%5Ctest.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%2520summative%2Ftrial%2Ftest.py%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%20summative%2Ftrial%2Ftest.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A15%2C%22character%22%3A16%7D%7D%5D%5D "Go to definition"): Builds and trains the DQN agent.
- [`test.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%20summative%2Ftrial%2Ftest.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\user\Desktop\codes\alu-machine_learning\unsupervised_learning\RL summative\trial\test.py"): Tests the trained DQN agent and visualizes the results.
- [`saved_weights/`](command:_github.copilot.openSymbolFromReferences?%5B%22saved_weights%2F%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cuser%5C%5CDesktop%5C%5Ccodes%5C%5Calu-machine_learning%5C%5Cunsupervised_learning%5C%5CRL%20summative%5C%5Ctrial%5C%5Ctest.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%2520summative%2Ftrial%2Ftest.py%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%20summative%2Ftrial%2Ftest.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A29%2C%22character%22%3A18%7D%7D%5D%5D "Go to definition"): Directory to store the trained model weights.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/shower-rl.git
    cd shower-rl
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training the Agent

To train the DQN agent, run the [`agent.py`](command:_github.copilot.openSymbolFromReferences?%5B%22agent.py%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cuser%5C%5CDesktop%5C%5Ccodes%5C%5Calu-machine_learning%5C%5Cunsupervised_learning%5C%5CRL%20summative%5C%5Ctrial%5C%5Ctest.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%2520summative%2Ftrial%2Ftest.py%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%20summative%2Ftrial%2Ftest.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A15%2C%22character%22%3A16%7D%7D%5D%5D "Go to definition") script:
```sh
python agent.py
```
This will train the agent and save the weights in the [`saved_weights/`](command:_github.copilot.openSymbolFromReferences?%5B%22saved_weights%2F%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cuser%5C%5CDesktop%5C%5Ccodes%5C%5Calu-machine_learning%5C%5Cunsupervised_learning%5C%5CRL%20summative%5C%5Ctrial%5C%5Ctest.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%2520summative%2Ftrial%2Ftest.py%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%20summative%2Ftrial%2Ftest.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A29%2C%22character%22%3A18%7D%7D%5D%5D "Go to definition") directory.

### Testing the Agent

To test the trained agent and visualize the results, run the [`test.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%20summative%2Ftrial%2Ftest.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\user\Desktop\codes\alu-machine_learning\unsupervised_learning\RL summative\trial\test.py") script:
```sh
python test.py
```
This will load the saved weights, run the agent for a specified number of episodes, and plot the total rewards per episode.

## Code Overview

### [`test.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fuser%2FDesktop%2Fcodes%2Falu-machine_learning%2Funsupervised_learning%2FRL%20summative%2Ftrial%2Ftest.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\user\Desktop\codes\alu-machine_learning\unsupervised_learning\RL summative\trial\test.py")

```python
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
        action = dqn.forward(state)  # Get action from the trained agent
        state, reward, done, info = env.step(action)  # Take the action
        env.render()
```

## Requirements

- Python 3.6+
- TensorFlow
- Keras-RL
- Matplotlib

Install the dependencies using:
```sh
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Keras-RL](https://github.com/keras-rl/keras-rl) for the reinforcement learning library.
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework.

---

Feel free to customize this README to better fit your project's specifics and add any additional sections as needed.