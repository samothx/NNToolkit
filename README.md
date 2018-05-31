# NNToolkit
A Neural Network Playground - neural network implementation in python based on numPy.
The code puts together the learnings from Coursea courses on Machine Learning and Deep Learning (https://www.coursera.org/learn/deep-neural-network).
It implements a multi-layered neural network and functions to train it or to evaluate it. The main motivation is to get experence with neural networks, their implementation and their opimization. 

## Architecture
I am rather new to python so some of the code is a little experimental in parts. Implementation is object oriented as follows:
- Class **Layer** - represents a layer of a neural network. The constructor optionally initializes a matrix W and vector b. These values are then reused and 
updated in each iteration. Layer objects are chained together to build a network. The central function is **process**. This function does the forward propagation and 
then calls the process function of the next layer. It expects a dictionary on return that contains the parameters computed by the final layer (see Terminal) and if back-propagation is enabled 
the derivative with respect to the activation A - dA handed back from the next layer.
With this it can (if configured to do so) compute the derivatives dW,dB,dA, update the matrix W,b hand hand back up the result and dA to the calling function.   
- Class **Terminal** - the last layer is a special layer implemented in class terminal. It computes the result vector Y_hat and if configured to do so the cost function and the derivatives with 
respect to the activation A. These values are handed back to the calling function and trvel back up the recursive chain.
- Class **Activation** - a set of classes, Sigmoid, TanH and ReLU that compute the derivatives of the activation function. Input is Z and dA the derivatives handed back from the next layer.

Further modules provide helper functions to create networks from parameters and to learn and evaluate a network.    

## Test Cases
Current test cases are rather simple self-generated input. I have managed to get the network to learn the following test case:
A matrix X of m randomly generated (column-) vectors with 5 elements (normalized random distribution with numpy.random.randn, multiplied by 20, added 10). 
A corresponding (row-)vector y that is 1 for every column of X that has a mean value greater 10.
The task for the network is to learn if a vectors mean value is greater 10 (1) or less (0).
The current implementation manages to find parameters that learn this projection with a training accuracy of 99.8% and 
a test accuracy of 99.5%.

I am still working to find a network that will solve my second test case: A fuzzy XOR.   
    

## Features 

This is work in progress. So far the network supports the following features:
- Supervised machine learning
- Currently supports only classification problems
. supports activation functions ReLU, Tanh, Sigmoid
- Free choice of topography
- Various hyper parameters. 
- Basic built in gradient descent
- Allows to supply parameters from outside to enable external mininmization algorithms. This feature is only a built in concept and has not been tested.


## Todos
Missing Features are:
- Support for normalization of input
- Regularization
- Support for more activation functions
- Support for linear regression Problems
 
I will update this project while the courses I am hearing progress and add more advanced features.     
 
## License ##
Feel free to fork this repository and use as you like.

## Disclaimer ##

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 
