import numpy as np

data = np.genfromtxt(
    "SingleLayerPerceptronDataset.csv",
    delimiter=",",
    skip_header=1,
    filling_values=0
)

X = data[:, 1:4]     
y = data[:, 4]      

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

print("\nPredictions:")
for i in range(len(X)):
    pred = np.dot(w, X[i]) + b
    print(f"Sample {i+1}: Prediction = {pred:.4f}, Actual = {y[i]}")