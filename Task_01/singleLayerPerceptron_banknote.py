import numpy as np

# =========================
# Load Dataset
# =========================
data = np.genfromtxt(
    "data_banknote_authentication.txt",
    delimiter=",",
    skip_header=0,
    filling_values=0
)

# Features and Labels
X = data[:, :-1]
y = data[:, -1]

# Normalize features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# =========================
# Initialize weights and bias
# =========================
num_features = X.shape[1]
w = np.random.randn(num_features)
b = 0

lr = 0.01
epochs = 1000

print("Initial Weights:", w)
print("Initial Bias:", b)

# =========================
# Training
# =========================
for epoch in range(epochs):
    total_error = 0

    for i in range(len(X)):
        x = X[i]
        target = y[i]

        # Weighted sum
        z = np.dot(w, x) + b

        # Step activation
        if z >= 0:
            prediction = 1
        else:
            prediction = 0

        # Error
        error = target - prediction
        total_error += abs(error)

        # Perceptron update rule
        w = w + lr * error * x
        b = b + lr * error

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Error = {total_error}")

    # Stop if perfectly classified
    if total_error == 0:
        print("\nTraining Converged!")
        break

# =========================
# Final Weights
# =========================
print("\nTraining Finished")
print("Final Weights:", w)
print("Final Bias:", b)

# =========================
# Accuracy
# =========================
correct = 0

for i in range(len(X)):
    z = np.dot(w, X[i]) + b

    if z >= 0:
        pred = 1
    else:
        pred = 0

    if pred == y[i]:
        correct += 1

accuracy = (correct / len(X)) * 100
print(f"Accuracy = {accuracy:.2f}%")