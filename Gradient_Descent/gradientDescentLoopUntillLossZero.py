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

threshold = 1e-6   # close enough to zero
step = 0

print("\nTraining Started...\n")

while True:
    step += 1

    # Forward pass
    a = np.dot(w, x) + b

    # Loss
    loss = (a - y) ** 2

    # Stop if loss is near zero
    if loss < threshold:
        break

    # Gradient
    grad = 2 * (a - y) * x

    # Update weights
    w = w - lr * grad

    print(f"Step {step}")
    print(f"Prediction = {a:.6f}")
    print(f"Loss = {loss:.8f}")
    print(f"Updated Weights = {w}")
    print("-" * 40)

print("\nTraining Finished!")
print(f"Total Steps = {step}")
print(f"Final Loss = {loss:.10f}")
print("Final Weights =", w)