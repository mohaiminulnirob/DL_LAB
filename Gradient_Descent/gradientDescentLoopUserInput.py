import numpy as np

# Take inputs from user
x = float(input("Enter x (input): "))
y = float(input("Enter y (target): "))
b = float(input("Enter b (bias): "))
w = float(input("Enter initial w (weight): "))
lr = float(input("Enter learning rate: "))
steps = int(input("Enter number of training steps: "))

print("\nTraining Started...\n")

for step in range(steps):
    # Forward pass
    a = w * x + b

    # Loss
    loss = (a - y) ** 2

    # Gradient
    grad = 2 * (a - y) * x

    # Weight update
    w = w - lr * grad

    print(f"Step {step+1}:")
    print(f" Prediction (a) = {a:.4f}")
    print(f" Loss = {loss:.6f}")
    print(f" Gradient = {grad:.4f}")
    print(f" Updated Weight = {w:.4f}")
    print("-" * 30)

print("\nTraining Finished!")
print(f"Final weight = {w:.4f}")