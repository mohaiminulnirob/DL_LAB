import numpy as np

# Fixed values
x = 2
y = 1
b = 0

# Initial weight
w = -1.0

# Learning rate
lr = 0.1

# Forward pass
a = w * x + b

# Loss
loss = (a - y) ** 2

# Gradient
grad = 2 * (a - y) * x

# Update
w_new = w - lr * grad

print("Prediction:", a)
print("Loss:", loss)
print("Gradient:", grad)
print("Updated weight:", w_new)