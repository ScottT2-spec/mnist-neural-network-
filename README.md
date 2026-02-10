# Neural Network from Scratch

A neural network that recognizes handwritten digits (0-9), built using only Python and NumPy. No TensorFlow, no PyTorch — every part of the network is written from scratch.

## What it does

Takes 28x28 pixel images of handwritten digits and predicts which number they are. Trained on the MNIST dataset (60,000 training images, 10,000 test images).

## How it works

- **Input layer:** 784 neurons (28x28 pixels flattened)
- **Hidden layer:** 64 neurons with sigmoid activation
- **Output layer:** 10 neurons (one per digit)

The network learns through backpropagation — it makes a prediction, checks how wrong it was, then adjusts its weights to be less wrong next time. After 20 passes through the full dataset, it reaches about 96% accuracy.

## Results

Final accuracy: **95.75%** on test data.

### Predictions on test digits

![Predictions](images/IMG_2228.jpeg)

### Confusion Matrix

![Confusion Matrix](images/IMG_2227.jpeg)

The confusion matrix shows where the network struggles. It's decent at most digits but sometimes confuses 4s and 9s (makes sense — they look similar if you write them a certain way).

## What I learned building this

- Forward propagation is just matrix multiplication through each layer, nothing more complicated than that
- Backpropagation looked scary at first but it's really just the chain rule — figure out how much each weight contributed to the error, then nudge it the other direction
- Normalizing inputs to 0-1 instead of 0-255 made a huge difference in training speed
- Small random weights (multiplied by 0.1) train way better than large ones because sigmoid saturates at extreme values
- Batch training (64 images at a time) is faster and more stable than updating after every single image

## The guide

I wrote a complete guide (`GUIDE.md`) that walks through everything step by step — NumPy basics, the math behind neural networks, and how to build one from scratch. Check it out if you want to understand how this works.

## Run it yourself

Open `neural_net_complete.py` in Google Colab or Jupyter Notebook and run all cells. Takes about 2 minutes to train.

## Built with

- Python 3
- NumPy (math operations)
- Matplotlib (visualization)
- Keras (only for loading the MNIST dataset — the actual network is from scratch)
