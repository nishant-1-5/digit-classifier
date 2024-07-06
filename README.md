# Digit Classification using Neural Network
This project involves building a digit classification model using a neural network. The model is trained on  [THE MNIST DATABASE]( http://yann.lecun.com/exdb/mnist/) of handwritten digits and can recognize handwritten digits.

## Dependencies
<ul><li>Matplotlib</li>
<li>Numpy</li>
<li>Tensorflow</li>
</ul>
You can install these packages via pip.

## Files
-   `X.npy`: The feature data for the digit dataset (i.e. the pixels).
-   `y.npy`: The labels for the digit dataset.
-   `DigRecog.py`: The main script to train and evaluate the digit classifier model.

## Description
This project uses a neural network to classify input digits between 0 and 9, It consists of the following layers:-
1. **Input Layer**: The input layer has 400 neurons, corresponding to the 20x20 pixel images flattened into a 400-dimensional vector.
2. **Hidden Layer 1**: The first hidden layer has 25 neurons and uses the ReLU activation function.
3.  **Hidden Layer 2**: The second hidden layer has 20 neurons and uses the ReLU activation function.
4. **Hidden Layer 3**: The third hidden layer has 15 neurons and uses the ReLU activation function.
5. **Output Layer**: The output layer has 10 neurons (one for each digit class) and uses a linear activation function. 
