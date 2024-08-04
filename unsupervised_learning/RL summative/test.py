# Define the step size to skip timesteps
step_size = 5  # Adjust this value as needed

for t in range(0, l, step_size):
    print(f"Processing time step {t}/{l}")
    action = agent.act(state)

    # sit
    next_state = getState(data, t + step_size, window_size + 1)
    reward = 0

    if action == 1:  # buy
        agent.inventory.append(data[t])
        print("Buy: " + formatPrice(data[t]))

    elif action == 2 and len(agent.inventory) > 0:  # sell
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price
        print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

    done = True if t >= l - step_size else False
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
        print("--------------------------------")
        print("Total Profit: " + formatPrice(total_profit))
        print("--------------------------------")