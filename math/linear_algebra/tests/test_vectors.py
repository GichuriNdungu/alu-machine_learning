#!/usr/bin/env python3
import numpy as np

# Define your matrix A
A = [[ 2, 2, 1 ],
  [ 2, 1, 3 ],
  [ 1, 3, 8 ]]
# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# The eigenvalues are stored in eigenvalues
print("Eigenvalues:", eigenvalues)

# The eigenvectors are stored in eigenvectors
print("Eigenvectors:")
print(eigenvectors)
