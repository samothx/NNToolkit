# NNToolkit
A Neural Network Playground - neural network implementation in python based on numPy.
The code puts together the learnings from Coursea courses on Machine Learning and Deep Learning 
(https://www.coursera.org/learn/deep-neural-network).
It implements a multi-layered neural network and functions to train it or to evaluate it. The main motivation is to get 
experience with neural networks, their implementation and their optimization. 

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
- Class **Activation** - a set of classes (Sigmoid, Softmax, TanH and ReLU) that compute the activation function and its 
derivatives da. They also compute the cost function (cost & dz) and y^. Cost and y^ are only implemented for Softmax and 
Sigmoid as thes values are only needed for the output layer. 

Further modules provide helper functions to create networks from parameters and to learn and evaluate a network.    

## Features 

This is work in progress. So far the network supports the following features:
- Supervised machine learning
- Currently supports only classification problems
- Supports activation functions ReLU, Tanh, Sigmoid, Softmax
- Automatic weight initialization with he / xavier
- Free choice of topography
- Various hyper parameters.
- L2 regularization.
- Dropout regularization.
- Adaptive learn rate.
- Momentum, RMSProp & Adam Gradient Descent
- Gradient checking.
- Graphical output of cost, train & cross validation error over iterations.
- Function to shuffle and split input into train,cross validation and test set 
- Basic built in gradient descent
- Allows to supply parameters from outside to enable external minimization algorithms.
- Save/load network parameters to/from file
- Plotting of cost function and errors over iterations


## Todos
Soon to be added missing Features are:
- Support convolutional networks
- Support for layerwise dropout
- Support for normalization of input
- Support for more activation functions
- Support for linear regression Problems
- Support for other optimization techniques for gradient descent.
- Support for parallel execution of mini-batch gradient descent (my very own strategy) 
 
I will update this project while the courses I am hearing progress and add more advanced features.     

## Example Code

Samples are in the sample sub directory. The handwriting example uses a subset of the MNIST handwritten digits example 
also called the Drosophila of image processing because it has been used in so many image recognition projects.   
The file samples/handwriting.py sets up a network and trains it to recognize the hand written digits. So far I have
best results in the range of 0.5% on the training set and 0-2% on the test set. The results on the test set are 
partially due to the small test set I am using. I was desperate to get as much training examples as possible as the 
dataset consists only of 5000 examples so my test set is only of size 50. So one sample not recognized immedeately gives 
me an error of 2%.

Its pretty easy to modify this so play around with the code in ```init_hand_writing() ```. The function call
```divide2sets(x_raw, y_class, 0.01, 0, True, True)``` does the splitting up of the dataset. First and second argument 
are the input X and labels Y, next parameters are the relative sizes of cross validation and test set (0.01 = 1%) and 
the last two boolean values are shuffle (does the data have to be 
shuffled before splitting) and transpose (does the data have to be transposed). 

So far the project is still missing a lot of documentation but I have commented the parameters used in handwriting. 
If you have seen the coursera lectures on Deep Learning you will recognize most of the parameter names.    

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
 
