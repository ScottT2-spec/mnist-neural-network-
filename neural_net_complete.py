import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load and prepare data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784) / 255.0
X_test = X_test.reshape(10000, 784) / 255.0
y_train_hot = np.eye(10)[y_train]
y_test_hot = np.eye(10)[y_test]

# Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1 - a)

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(784, 64) * 0.1
b1 = np.zeros((1, 64))
W2 = np.random.randn(64, 10) * 0.1
b2 = np.zeros((1, 10))

# Training
learning_rate = 0.5
epochs = 20
batch_size = 64

for epoch in range(epochs):
    indices = np.random.permutation(60000)
    X_shuffled = X_train[indices]
    y_shuffled = y_train_hot[indices]
    for i in range(0, 60000, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        hidden = sigmoid(X_batch.dot(W1) + b1)
        output = sigmoid(hidden.dot(W2) + b2)
        error = output - y_batch
        d_output = error * sigmoid_derivative(output)
        d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden)
        W2 = W2 - learning_rate * hidden.T.dot(d_output) / batch_size
        b2 = b2 - learning_rate * d_output.mean(axis=0, keepdims=True)
        W1 = W1 - learning_rate * X_batch.T.dot(d_hidden) / batch_size
        b1 = b1 - learning_rate * d_hidden.mean(axis=0, keepdims=True)
    hidden_test = sigmoid(X_test.dot(W1) + b1)
    output_test = sigmoid(hidden_test.dot(W2) + b2)
    predictions = np.argmax(output_test, axis=1)
    accuracy = np.mean(predictions == y_test) * 100
    print("Epoch", epoch + 1, "- Accuracy:", round(accuracy, 2), "%")

print("Training complete! Final accuracy:", round(accuracy, 2), "%")

# Visualize predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, ax in enumerate(axes.flat):
    test_image = X_test[idx].reshape(1, 784)
    hidden = sigmoid(test_image.dot(W1) + b1)
    output = sigmoid(hidden.dot(W2) + b2)
    prediction = np.argmax(output)
    actual = y_test[idx]
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    color = "green" if prediction == actual else "red"
    ax.set_title("Pred: " + str(prediction) + " Real: " + str(actual), color=color)
    ax.axis("off")
plt.tight_layout()
plt.show()

# Confusion matrix
matrix = np.zeros((10, 10), dtype=int)
for real, pred in zip(y_test, predictions):
    matrix[real][pred] += 1

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(matrix, cmap="Blues")
for i in range(10):
    for j in range(10):
        color = "white" if matrix[i][j] > 500 else "black"
        ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color=color, fontsize=10)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
plt.tight_layout()
plt.show()

for digit in range(10):
    mask = y_test == digit
    digit_acc = np.mean(predictions[mask] == digit) * 100
    print("Digit", digit, "accuracy:", round(digit_acc, 2), "%")
