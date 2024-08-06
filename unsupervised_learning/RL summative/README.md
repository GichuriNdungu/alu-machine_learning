# Stock Trading Algorithm

This project implements a stock trading algorithm using a neural network model and reinforcement learning. The algorithm is designed to make buy and sell decisions based on historical stock data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Reinforcement Learning](#reinforcement-learning)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/RL summative.git
    cd your-repo-name
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Model Training

To train the model, run:
```sh
python train.py [stock] [window_size] [episodes]
```
- `stock`: The stock symbol to train on (e.g., `AAPL`).
- `window_size`: The size of the window for the state representation.
- `episodes`: The number of training episodes.

### Evaluation

To evaluate the model, run:
```sh
python evaluate.py [stock] [model]
```
- `stock`: The stock symbol to evaluate on (e.g., `AAPL`).
- `model`: The name of the trained model file (e.g., `model_ep200`).

### Visualization

The evaluation script will generate a plot showing the stock prices along with buy and sell signals.

## Project Structure

- `agent/`: Contains the agent implementation.
- `models/`: Directory where trained models are saved.
- `data/`: Directory for storing stock data.
- `train.py`: Script for training the model.
- `evaluate.py`: Script for evaluating the model.
- `functions.py`: Utility functions for data processing and state representation.

## Reinforcement Learning

This project uses reinforcement learning to train the trading algorithm. Specifically, it employs a Deep Q-Learning (DQN) approach. Here’s a brief overview of the process:

1. **Agent**: The agent interacts with the environment (stock market) and makes decisions (buy, sell, hold).
2. **State**: The state is represented by a window of historical stock prices.
3. **Action**: The possible actions are buy, sell, or hold.
4. **Reward**: The reward is the profit or loss resulting from the action taken.
5. **Q-Learning**: The agent uses Q-learning to update its knowledge based on the rewards received, aiming to maximize cumulative profit.

## Model Training

The model is trained using a deep Q-learning algorithm. The agent learns to make buy, sell, or hold decisions based on the state representation of the stock prices.

## Evaluation

The evaluation script tests the trained model on historical stock data and visualizes the trading decisions.

## Visualization

The visualization includes:
- Stock prices over time.
- Buy signals (green triangles).
- Sell signals (red inverted triangles).
- Cumulative profit over time.

## Project live demonstration

Please find a live tutorial of this project [here]
## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
possible areas of improvement include:
- increasing the training epochs for a more accurate model. The current model is trained on 200 epochs, tuning this to 1000 would greatly improve the quality of the model.
- Tracking profits based on the decisions made by the model.
- This model is limited to short-term stock trading decisions and hence its not very good for at long-term stock trading. Possible contributions could help advance this model to suit long-term stock trading.

## Authors

- Martin Ndungu [m.ndungu@alustudent.com]