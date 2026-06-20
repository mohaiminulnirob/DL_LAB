import numpy as np

x = 2
y = 1
b = 0
w = -1.0
lr = 0.1

for step in range(15):
    a = w * x + b
    loss = (a - y) ** 2
    grad = 2 * (a - y) * x
    w = w - lr * grad

    print(f"Step {step+1}: w={w:.4f}, loss={loss:.6f}")