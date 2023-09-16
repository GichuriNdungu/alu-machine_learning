import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# create a list of labels for the categories

categories = ['Farrah', 'Fred', 'Felicia']
# generate a list of colors
colors = ['red', 'Yellow', '#ff8000', '#ffe5b4']
# create a list of quantities
quantities = ['apples', 'bananas', 'oranges', 'peaches']
# create an array to hold the bottom values for each bar
# np.zeros creates a list of 0's

bottom = np.zeros(len(categories))

# create a figure and an axis

fig, ax = plt.subplots()

# Loop through each row in the array (each row represents the quantities of the fruits)
for fruit_no in range(len(fruit)):
    ax.bar(categories, fruit[fruit_no], width=0.5,
           label=quantities[fruit_no], bottom=bottom, color=colors[fruit_no])
    bottom += fruit[fruit_no]
# add a label and a legend
ax.set_xlabel('categories')
ax.set_ylabel('Quantity')
ax.set_title('Number of Fruit per Person')
ax.set_ylim(0, 80)
ax.legend()

# show plot
plt.show()
