import numpy as np

# -----------------------------
# Step 1: Prepare Dataset
# Features:
# [Hours Study, Sleep Hours, Attendance]
# Target:
# 1 = Pass, 0 = Fail
# -----------------------------
X = np.array([
    [5, 7, 80],
    [2, 5, 50],
    [8, 6, 90],
    [1, 4, 30]
], dtype=float)

y = np.array([1, 0, 1, 0], dtype=float)

# -----------------------------
# Step 2: Initialize weights
# Since we have 3 features,
# we need 3 weights
# -----------------------------
w = np.random.randn(3)
b = 0.0

lr = 0.0001
epochs = 1000

print("Initial Weights:", w)
print("Initial Bias:", b)

# -----------------------------
# Step 3: Training Loop
# -----------------------------
for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):
        x = X[i]
        target = y[i]

        # Forward pass
        prediction = np.dot(w, x) + b

        # Loss
        loss = (prediction - target) ** 2
        total_loss += loss

        # Gradient calculation
        grad_w = 2 * (prediction - target) * x
        grad_b = 2 * (prediction - target)

        # Update weights and bias
        w = w - lr * grad_w
        b = b - lr * grad_b

    # Print every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss:.6f}")

# -----------------------------
# Step 4: Final Results
# -----------------------------
print("\nTraining Finished")
print("Final Weights:", w)
print("Final Bias:", b)

# -----------------------------
# Step 5: Predictions
# -----------------------------
print("\nPredictions on Training Data:")

for i in range(len(X)):
    pred = np.dot(w, X[i]) + b
    print(f"Sample {i+1}: Prediction = {pred:.4f}, Actual = {y[i]}")