import numpy as np

# Inputs
x = np.array([1.0, 0.5, -0.8])

# Weights
w = np.array([0.6, -0.9, 0.4])

# Bias
b = 0.2

# Forward pass
z = np.dot(x, w) + b

# Sigmoid activation
a = 1 / (1 + np.exp(-z))

print("z =", z)
print("a =", a)