import numpy as np

# Number of inputs
n = int(input("Enter number of inputs: "))

# Input vector
x = np.array(list(map(float, input(f"Enter {n} input values separated by space: ").split())))

# Weight vector
w = np.array(list(map(float, input(f"Enter {n} weight values separated by space: ").split())))

# Other parameters
b = float(input("Enter bias: "))
y = float(input("Enter target value: "))
lr = float(input("Enter learning rate: "))
steps = int(input("Enter number of training steps: "))

print("\nTraining Started...\n")

for step in range(steps):
    # Forward pass
    a = np.dot(w, x) + b

    # Loss
    loss = (a - y) ** 2

    # Gradient for all weights
    grad = 2 * (a - y) * x

    # Update weights
    w = w - lr * grad

    print(f"Step {step+1}")
    print(f"Prediction = {a:.4f}")
    print(f"Loss = {loss:.6f}")
    print(f"Gradient = {grad}")
    print(f"Updated Weights = {w}")
    print("-" * 40)

print("\nTraining Finished!")
print("Final Weights:", w)