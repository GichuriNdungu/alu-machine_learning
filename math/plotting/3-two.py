#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)
plt.xlabel('Time(Years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')
plt.ylim(0, 1)
plt.plot(x, y1, color='red', linestyle='dotted', label='C-14')
plt.plot(x, y2, color='green', label='Ra-226')
plt.legend(loc='upper right')
plt.show()
