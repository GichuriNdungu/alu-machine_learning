import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
	print ("Usage: python evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

prices = []
actions = []
profits = []

for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0
	
	prices.append(data[t])
	actions.append(action)

	if action == 1: # buy
		agent.inventory.append(data[t])
		print ("Buy: " + formatPrice(data[t]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		profits.append(total_profit)
		print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print ("--------------------------------")
		print (stock_name + " Total Profit: " + formatPrice(total_profit))
		print ("--------------------------------")

plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(prices, label='Stock Price')
buy_signals = [prices[i] if actions[i] == 1 else None for i in range(len(prices))]
sell_signals = [prices[i] if actions[i] == 2 else None for i in range(len(prices))]
plt.plot(buy_signals, 'g^', label='Buy Signal', markersize=5)
plt.plot(sell_signals, 'rv', label='Sell Signal', markersize=5)
plt.title(f'{stock_name} Trading Visualization')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()