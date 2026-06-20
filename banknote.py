import numpy as np

data = np.genfromtxt(
    "data_banknote_authentication.txt",
    delimiter=",",
    skip_header=1,
    filling_values=0
)

X = data[:, :-1]   
y = data[:, -1]     
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

num_features = X.shape[1]
w = np.random.randn(num_features)

b = 0

lr = 0.01
epochs = 1000

print("Initial Weights:", w)
print("Bias:", b)

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):
        x = X[i]
        target = y[i]

        prediction = np.dot(w, x) + b

        loss = (prediction - target) ** 2
        total_loss += loss

        grad_w = 2 * (prediction - target) * x

        w = w - lr * grad_w

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss:.6f}")

print("\nTraining Finished")
print("Final Weights:", w)
print("Bias:", b)

correct = 0

for i in range(len(X)):
    pred = np.dot(w, X[i]) + b

    # Convert raw output to class label
    if pred >= 0.5:
        predicted_class = 1
    else:
        predicted_class = 0

    # print(f"Sample {i+1}: Prediction={pred:.4f}, Predicted Class={predicted_class}, Actual={int(y[i])}")

    if predicted_class == y[i]:
        correct += 1

accuracy = (correct / len(X)) * 100

print(f"Accuracy = {accuracy:.2f}%")