# NNToolkit
A Neural Network Playground - neural network implementation in python based on numPy.
The code puts together the learnings from Coursea courses on Machine Learning and Deep Learning (https://www.coursera.org/learn/deep-neural-network).
It implements a multi-layered neural network and functions to train it or to evaluate it. The main motivation is to get experence with neural networks, their implementation and their optimization. 

## Architecture
I am rather new to python so some of the code might be a little clumsy. 
Implementation is object oriented as follows:
- Class **Layer** - represents a layer of a neural network. The constructor optionally initializes a 
matrix W and vector b. These values are then reused and updated in each iteration. 
Layer objects are chained together to build a network. The central function is **process**. 
This function does the forward propagation and then calls the process function of the next layer. 
It expects a dictionary on return that contains the parameters computed by the final layer 
(see Terminal) and if back-propagation is enabled the derivative with respect to the activation A - dA 
handed back from the next layer.
With this it can (if configured to do so) compute the derivatives dW,dB,dA, update the matrix W,b and 
hand back up the result and dA to the calling function.   
- Class **Terminal** - the last layer is a special layer implemented in class terminal. It computes the 
result vector Y_hat and if configured to do so the cost function and the derivatives with respect to 
the activation A. These values are handed back to the calling function and travel back up the recursive chain. 
The Terminal class will be replaced by a set of classes that represent other result types. The current implementation 
only implements the functionality for classification problems.   
- Class **Activation** - a set of classes, Sigmoid, TanH and ReLU that compute the activation function and its derivatives. 
Input is Z and dA the derivatives handed back from the next layer. Each layer uses an 
instance of one of these classes to compute the activation function and the derivatives.

Further modules provide helper functions to create networks from parameters and to learn and evaluate a network.    

## Features 

This is work in progress. So far the network supports the following features:
- Supervised machine learning
- Currently supports only classification problems
. supports activation functions ReLU, Tanh, Sigmoid
- Free choice of topography
- Various hyper parameters.
- L2 regularization.
- Adaptive learn rate.
- Function to shuffle and split input into train,cross validation and test set 
- Basic built in gradient descent
- Allows to supply parameters from outside to enable external minimization algorithms. This feature is only a built in concept and has not been tested.
- Save/load network parameters to/from file


## Todos
Soon to be added missing Features are:
- Support for normalization of input
- Support for more activation functions
- Support for linear regression Problems
 
I will update this project while the courses I am hearing progress and add more advanced features.     

## Example Code

Samples are in the sample sub directory. The handwriting example uses input data from a previous course. 
I am not sure about copyright copyright on that data, so I have not checked it in, sorry. I will try to find out if I can do so. 

Send me a message for working parameter sets.   

## Test Cases
Current test cases are rather simple self-generated input. I have managed to get the network to learn the 
following test case:
A matrix X of m randomly generated (column-) vectors with 5 elements (normalized random distribution with 
numpy.random.randn, multiplied by 20, added 10). 
A corresponding (row-)vector y that is 1 for every column of X that has a mean value greater 10.
The task for the network is to learn if a vectors mean value is greater 10 (1) or less (0).
The current implementation manages to find parameters that learn this projection with a training accuracy 
of 99.8% and a test accuracy of 99.5%. This is the interesting part as it gives you hands on experience in 
how to actually find the right parameters to make a neural network converge and find a good solution.  

I am still working to find a network that will solve my second test case: A fuzzy XOR. I started with a 'digital' XOR 
meaning 16 Vectors of 4 values 1 or 0. Basically these vectors were the binary representation of the numbers 
0-15. From these vectors I computed a XOR by going through the vector top down as follows:
```python

def rand_data(n):
    m = 2**n
    X = np.zeros((n,m))
    Y = np.zeros((1,m))

    for i in range(0,m):
        tmp = 0
        for j in range(0,n):
            X[j,i] = (i & (1 << j)) > 0
            if j == 0:
                tmp = X[j,i]
            else:
                tmp = tmp != X[j,i]
        Y[0,i] = tmp   
    Y = Y * 1
    X = X * 1
    return X,Y

```    
Unfortunately so far I was not able to find a network willing to learn this function.
Meanwhile I have fallen back to the following fuzzy implementation that performs only little 
better but allows me to create bigger datasets than m = 2^n:
```python
def rand_data(n,m):
    x = np.random.rand(n,m)
    x_dig = (x > 0.5)
    y = np.zeros((1,m))
    for i in range(0, m):
        y_tmp = x_dig[0, i]
        for j in range(1, n):
            y_tmp = y_tmp != x_dig[j, i]
        y[0, i] = y_tmp * 1
    return x,y
``` 

 
## License ##
Feel free to fork this repository and use as you like.

## Disclaimer ##

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 
