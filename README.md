# Neural Network from Scratch

A neural network that recognizes handwritten digits (0-9), built using only Python and NumPy. No TensorFlow, no PyTorch — every part of the network is written from scratch.

## What it does

Takes 28x28 pixel images of handwritten digits and predicts which number they are. Trained on the MNIST dataset (60,000 training images, 10,000 test images).

## How it works

- **Input layer:** 784 neurons (28x28 pixels flattened)
- **Hidden layer:** 64 neurons with sigmoid activation
- **Output layer:** 10 neurons (one per digit)

The network learns through backpropagation — it makes a prediction, checks how wrong it was, then adjusts its weights to be less wrong next time. After 20 passes through the full dataset, it reaches ~96% accuracy.

## Results

Final accuracy: **95.75%** on test data.

### Predictions on test digits

![Predictions](IMG_2228.jpeg)

### Confusion Matrix

![Confusion Matrix](IMG_2227.jpeg)

## What I learned

- How forward propagation moves data through a network
- How backpropagation calculates which weights caused errors
- How gradient descent adjusts weights to reduce error over time
- Why we normalize inputs (0-1 range instead of 0-255)
- Why one-hot encoding is needed for classification

## Run it yourself

Open `neural_net_complete.py` in Google Colab or Jupyter Notebook and run all cells.

## Built with

- Python 3
- NumPy (math operations)
- Matplotlib (visualization)
- Keras (only for loading the MNIST dataset)
