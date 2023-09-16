#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))
plt.suptitle('All in One', fontsize='x-small')
# o-line
plt.subplot(3,2,1)
y0 = np.arange(0, 11) ** 3
x = np.arange(0, 11)
plt.plot(x, y0, color='red')
#1-scatter
plt.subplot(3,2,2)
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180
plt.xlabel('Height (in)', fontsize='x-small')
plt.ylabel('Weight (lbs)', fontsize='x-small')
plt.title('Men\'s Height vs Weight', fontsize='x-small')
plt.scatter(x1, y1, color='magenta')

#2-change_scale
plt.subplot(3,2,3)
x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)
plt.xlabel('Time(Years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of C-14', fontsize='x-small')
plt.yscale('log')
plt.xlim(0, np.max(x2))
plt.ylim(np.min(y2), 1.0)
plt.plot(x2, y2, color='blue')
#3-two
plt.subplot(3,2,4)
x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)
plt.xlabel('Time(Years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of C-14', fontsize='x-small')
plt.ylim(0, 1)
plt.plot(x3, y31, color='red', linestyle='dotted', label='C-14')
plt.plot(x3, y32, color='green', label='Ra-226')
plt.legend(loc='upper right')

#4-frequency
plt.subplot(3,2,(5,6))
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
bin_edges = np.arange(min(student_grades), max(student_grades) + 11, 10)
plt.hist(student_grades, bins=bin_edges, edgecolor='black')
plt.xlabel('Grades', fontsize='x-small')
plt.ylabel('Number of Students', fontsize='x-small')
plt.title('Project A', fontsize='x-small')
#adjst the layout

plt.tight_layout()
#show all subplots

plt.show()
