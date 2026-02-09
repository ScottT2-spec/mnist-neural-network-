# 🧠 Build Your Own Neural Network from Scratch
### A Complete Guide — No Libraries, Just Python + NumPy

---

## PART 1: Python Topics You MUST Know

### Level 1 — Basics (You probably know these)
- **Variables & types:** `int`, `float`, `str`, `bool`
- **Math operators:** `+`, `-`, `*`, `/`, `**` (power), `%` (modulo)
- **Print & f-strings:** `print(f"Accuracy: {acc:.2f}%")`
- **If/elif/else:** conditional logic
- **Loops:** `for i in range(10)`, `while`, `break`, `continue`
- **Functions:** `def`, parameters, `return`, default arguments
- **Lists:** creating, indexing, slicing, `.append()`, list comprehension

### Level 2 — Intermediate (Study these harder)
- **Dictionaries:** `{}`, `.keys()`, `.values()`, `.items()`
- **Tuples:** immutable sequences, tuple unpacking: `a, b = (1, 2)`
- **String formatting:** f-strings, `.format()`
- **List comprehension:** `[x**2 for x in range(10) if x > 3]`
- **Nested loops:** loops inside loops (needed for batches + epochs)
- **Error handling:** `try/except` (useful, not critical for NN)

### Level 3 — NumPy (THIS IS THE BIG ONE 🔥)

NumPy is what makes neural networks possible in Python. Without it, everything would be 1000x slower. Study these until they're second nature:

#### Creating arrays
```python
import numpy as np

a = np.array([1, 2, 3])           # 1D array (vector)
b = np.array([[1,2],[3,4]])        # 2D array (matrix)
c = np.zeros((3, 4))              # 3x4 matrix of zeros
d = np.ones((2, 5))               # 2x5 matrix of ones
e = np.random.randn(3, 3)        # 3x3 random numbers (normal distribution)
f = np.eye(5)                     # 5x5 identity matrix (1s on diagonal)
```

#### Shape and reshape
```python
a = np.array([[1,2,3],[4,5,6]])   # Shape: (2, 3) — 2 rows, 3 columns
a.shape                            # Returns (2, 3)
a.reshape(3, 2)                    # Reshape to 3 rows, 2 columns
a.reshape(6)                       # Flatten to 1D: [1,2,3,4,5,6]
a.T                                # Transpose: rows become columns
```
**Why it matters:** Images are 28x28, we reshape to 784. Weights need specific shapes to multiply.

#### Indexing and slicing
```python
a = np.array([10, 20, 30, 40, 50])
a[0]        # 10 (first element)
a[-1]       # 50 (last element)
a[1:4]      # [20, 30, 40] (index 1 to 3)
a[:3]       # [10, 20, 30] (first 3)

# 2D indexing
b = np.array([[1,2,3],[4,5,6],[7,8,9]])
b[0]        # [1, 2, 3] (first row)
b[:, 1]     # [2, 5, 8] (second column)
b[0:2, :]   # First 2 rows, all columns
```
**Why it matters:** You constantly slice batches, select columns, index into matrices.

#### Math operations (element-wise)
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b       # [5, 7, 9]    — add matching elements
a * b       # [4, 10, 18]  — multiply matching elements
a ** 2      # [1, 4, 9]    — square each element
1 / a       # [1.0, 0.5, 0.33] — divide 1 by each element
np.exp(a)   # [2.7, 7.4, 20.1] — e raised to each element
```
**Why it matters:** Activation functions, error calculation, weight updates — ALL element-wise.

#### ⭐ Matrix multiplication (dot product) — MOST IMPORTANT
```python
# This is the CORE of neural networks
A = np.array([[1,2],[3,4]])     # Shape: (2, 2)
B = np.array([[5,6],[7,8]])     # Shape: (2, 2)

C = A.dot(B)        # Matrix multiply
# OR
C = np.dot(A, B)    # Same thing
# OR  
C = A @ B           # Same thing (Python 3.5+)

# Rule: (m, n) dot (n, p) = (m, p)
# The inner dimensions MUST match
# (784, 64) dot (64, 10) = (784, 10) ✅
# (784, 64) dot (10, 64) = ERROR ❌
```

**How matrix multiply works (conceptually):**
```
Input (1 image, 784 pixels) × Weights (784 connections to 64 neurons) = Hidden (1 image, 64 values)

Each hidden neuron = sum of (every pixel × its weight to that neuron)

Neuron 1 = pixel1×w1 + pixel2×w2 + ... + pixel784×w784
Neuron 2 = pixel1×w1 + pixel2×w2 + ... + pixel784×w784
...
Neuron 64 = pixel1×w1 + pixel2×w2 + ... + pixel784×w784
```
**This is literally what `.dot()` computes in one line.**

#### Broadcasting
```python
a = np.array([[1,2,3],
              [4,5,6]])        # Shape: (2, 3)
b = np.array([10, 20, 30])    # Shape: (3,)

a + b  # [[11,22,33],[14,25,36]]
# NumPy automatically adds b to EACH ROW of a
```
**Why it matters:** When you do `X_batch.dot(W1) + b1`, the bias `b1` is shape (1, 64) but gets added to every row in the batch. That's broadcasting.

#### Axis operations
```python
a = np.array([[1,2,3],
              [4,5,6]])

np.sum(a)           # 21 (sum everything)
np.sum(a, axis=0)   # [5, 7, 9] (sum down columns)
np.sum(a, axis=1)   # [6, 15] (sum across rows)
np.mean(a, axis=0)  # [2.5, 3.5, 4.5] (average down columns)
np.argmax(a, axis=1) # [2, 2] (index of largest in each row)
```
**Why it matters:** 
- `axis=0` = average across batch (for weight updates)
- `argmax(axis=1)` = which digit has highest probability (prediction)

#### Random and permutation
```python
np.random.seed(42)              # Reproducible results
np.random.randn(3, 4)          # Random matrix, normal distribution
np.random.permutation(100)     # Shuffled array [0-99]
```
**Why it matters:** Initialize weights randomly, shuffle data each epoch.

### Level 4 — Matplotlib (For visualization)
```python
import matplotlib.pyplot as plt

# Show an image
plt.imshow(image_array, cmap="gray")
plt.title("Digit: 5")
plt.show()

# Plot a line graph (loss over time)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Subplots (multiple images)
fig, axes = plt.subplots(2, 5)
for ax in axes.flat:
    ax.imshow(some_image, cmap="gray")
plt.show()
```

---

## PART 2: The Math You Need (Don't Panic)

You don't need to be a math genius. You need to understand these concepts:

### 1. Weighted Sum
```
output = input1×weight1 + input2×weight2 + ... + bias
```
That's it. Every neuron does this. Matrix multiplication does it for ALL neurons at once.

### 2. Activation Function (Sigmoid)
```
sigmoid(x) = 1 / (1 + e^(-x))
```
- Input: any number (-∞ to +∞)
- Output: number between 0 and 1
- Big positive → close to 1 (neuron fires)
- Big negative → close to 0 (neuron doesn't fire)
- Zero → 0.5 (undecided)

### 3. Derivative (Rate of Change)
- Don't need calculus. Just know: derivative tells you "which direction to adjust"
- Sigmoid derivative: `a × (1 - a)` where a = sigmoid output
- If a = 0.5 → derivative = 0.25 (learn a lot, neuron is uncertain)
- If a = 0.99 → derivative = 0.0099 (barely learn, neuron is already confident)

### 4. Gradient Descent
Imagine you're blindfolded on a hill, trying to reach the bottom:
- Feel the slope under your feet (= calculate gradient)
- Take a step downhill (= update weights)
- Repeat until flat ground (= minimum error)

`new_weight = old_weight - learning_rate × gradient`

---

## PART 3: Build a Neural Network Step by Step

### Step 1: Single Neuron
Start with ONE neuron. One input, one output.
```python
import numpy as np

# One neuron: 1 input → 1 output
weight = 0.5
bias = 0.0
input_val = 2.0

# Forward pass
output = input_val * weight + bias   # 2.0 * 0.5 + 0 = 1.0

# Apply activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

activated = sigmoid(output)   # sigmoid(1.0) = 0.73
print(f"Output: {activated}")
```
**What you learned:** A neuron = multiply input by weight, add bias, squash with sigmoid.

### Step 2: Multiple Inputs, One Neuron
```python
# 3 inputs → 1 neuron
inputs = np.array([1.0, 2.0, 3.0])
weights = np.array([0.2, 0.8, -0.5])
bias = 0.1

z = np.dot(inputs, weights) + bias  # 1×0.2 + 2×0.8 + 3×(-0.5) + 0.1 = 0.4
output = sigmoid(z)                  # sigmoid(0.4) = 0.60
```
**What you learned:** Multiple inputs → dot product → one output.

### Step 3: A Full Layer (Multiple Neurons)
```python
# 3 inputs → 4 neurons
inputs = np.array([1.0, 2.0, 3.0])           # Shape: (3,)
weights = np.random.randn(3, 4) * 0.1         # Shape: (3, 4)
biases = np.zeros((1, 4))                      # Shape: (1, 4)

z = inputs.dot(weights) + biases               # Shape: (4,) — one value per neuron
output = sigmoid(z)                             # Shape: (4,)
print(f"Layer output: {output}")
```
**What you learned:** One layer = dot product gives multiple outputs simultaneously.

### Step 4: Two Layers (Full Network!)
```python
# 784 inputs → 64 hidden → 10 outputs
X = np.random.randn(1, 784)                    # One fake image
W1 = np.random.randn(784, 64) * 0.1
b1 = np.zeros((1, 64))
W2 = np.random.randn(64, 10) * 0.1
b2 = np.zeros((1, 10))

# Forward pass
hidden = sigmoid(X.dot(W1) + b1)               # (1,784)×(784,64) = (1,64)
output = sigmoid(hidden.dot(W2) + b2)           # (1,64)×(64,10) = (1,10)

prediction = np.argmax(output)                   # Which digit (0-9) scored highest?
print(f"Predicted digit: {prediction}")
print(f"Confidence: {output}")
```
**What you learned:** Stack layers. Output of layer 1 = input to layer 2.

### Step 5: Calculate Error
```python
# The answer was digit "3"
target = np.eye(10)[3]  # [0,0,0,1,0,0,0,0,0,0]

error = output - target
# If output[3] = 0.2, error[3] = 0.2 - 1 = -0.8 (needs to be MUCH higher)
# If output[7] = 0.6, error[7] = 0.6 - 0 = 0.6 (needs to be lower)
```
**What you learned:** Error = guess - answer. Positive = too high, negative = too low.

### Step 6: Backpropagation
```python
def sigmoid_derivative(a):
    return a * (1 - a)

# How much to adjust output layer
d_output = error * sigmoid_derivative(output)

# How much to adjust hidden layer (blame flows backward)
d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden)
```
**What you learned:** Errors flow backward. Each layer's adjustment depends on the next layer's error.

### Step 7: Update Weights
```python
learning_rate = 0.5

W2 = W2 - learning_rate * hidden.T.dot(d_output)
b2 = b2 - learning_rate * d_output
W1 = W1 - learning_rate * X.T.dot(d_hidden)
b1 = b1 - learning_rate * d_hidden
```
**What you learned:** `new = old - learning_rate × gradient`. Gradient = how to change to reduce error.

### Step 8: Full Training Loop
```python
# Combine everything!
for epoch in range(20):
    for i in range(0, len(X_train), 64):
        batch = X_train[i:i+64]
        labels = y_train_hot[i:i+64]
        
        # Forward
        hidden = sigmoid(batch.dot(W1) + b1)
        output = sigmoid(hidden.dot(W2) + b2)
        
        # Error + Backprop
        error = output - labels
        d_output = error * sigmoid_derivative(output)
        d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden)
        
        # Update
        W2 -= 0.5 * hidden.T.dot(d_output) / 64
        b2 -= 0.5 * d_output.mean(axis=0, keepdims=True)
        W1 -= 0.5 * batch.T.dot(d_hidden) / 64
        b1 -= 0.5 * d_hidden.mean(axis=0, keepdims=True)
    
    # Check accuracy
    test_hidden = sigmoid(X_test.dot(W1) + b1)
    test_output = sigmoid(test_hidden.dot(W2) + b2)
    accuracy = np.mean(np.argmax(test_output, axis=1) == y_test) * 100
    print(f"Epoch {epoch+1}: {accuracy:.2f}%")
```

---

## PART 4: Cheat Sheet — The Complete Flow

```
┌─────────────────────────────────────────────┐
│  NEURAL NETWORK IN 9 LINES                  │
├─────────────────────────────────────────────┤
│  1. Load data, normalize, one-hot encode    │
│  2. Initialize weights (small random)       │
│  3. Forward: hidden = sigmoid(X·W1 + b1)    │
│  4. Forward: output = sigmoid(H·W2 + b2)    │
│  5. Error = output - target                 │
│  6. d_output = error × sigmoid'(output)     │
│  7. d_hidden = d_output·W2ᵀ × sigmoid'(H)  │
│  8. Update: W = W - lr × gradient           │
│  9. Repeat for N epochs                     │
└─────────────────────────────────────────────┘
```

## PART 5: Practice Exercises

After studying this guide, try these WITHOUT looking at the code:

**Exercise 1:** Create a NumPy array of shape (5, 3), multiply it by a (3, 2) matrix. What shape is the result?

**Exercise 2:** Write the sigmoid function from memory. Test it with inputs -10, 0, 10.

**Exercise 3:** Write a single neuron with 4 inputs. Do forward pass manually.

**Exercise 4:** Write a full 2-layer network (any size). Just the forward pass — no training.

**Exercise 5:** Add backpropagation and training to Exercise 4.

**Exercise 6:** Load MNIST, train your network, get >90% accuracy.

If you can do Exercise 6 from a blank file without looking at any reference, you're ready for MBZUAI. 🐺

---

*Feb 2026*
