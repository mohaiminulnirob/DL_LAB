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
b = 0.0

lr = 0.01
epochs = 1000

print("Initial Weights:", w)
print("Initial Bias:", b)

# =========================
# Sigmoid Function
# =========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# =========================
# Training
# =========================
for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):
        x = X[i]
        target = y[i]

        # -------------------------
        # Forward Pass
        # -------------------------
        z = np.dot(w, x) + b
        prediction = sigmoid(z)

        # -------------------------
        # BCE Loss
        # -------------------------
        loss = -(target * np.log(prediction + 1e-8) +
                 (1 - target) * np.log(1 - prediction + 1e-8))
        total_loss += loss

        # -------------------------
        # Gradient for BCE + Sigmoid
        # delta = prediction - target
        # -------------------------
        grad_w = (prediction - target) * x
        grad_b = prediction - target

        # -------------------------
        # Update
        # -------------------------
        w = w - lr * grad_w
        b = b - lr * grad_b

        # =========================
        # MSE Version (Commented)
        # =========================
        # mse_loss = (prediction - target) ** 2
        # grad_w = 2 * (prediction - target) * prediction * (1 - prediction) * x
        # grad_b = 2 * (prediction - target) * prediction * (1 - prediction)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, BCE Loss = {total_loss:.6f}")

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
    pred_prob = sigmoid(z)

    if pred_prob >= 0.5:
        pred = 1
    else:
        pred = 0

    if pred == y[i]:
        correct += 1

accuracy = (correct / len(X)) * 100
print(f"Accuracy = {accuracy:.2f}%")