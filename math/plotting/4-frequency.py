import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Specify the bin edges to create bins every 10 units
bin_edges = np.arange(min(student_grades), max(student_grades) + 11, 10)

plt.hist(student_grades, bins=bin_edges, edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')

plt.show()
