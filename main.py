import numpy as np
import NNLayer.create_network as cn
import NNLayer.activation as act
from NNLayer.util import print_matrix


# TODO: adaptive learning rate
# TODO: input normalization
# TODO: Fuzzy XOR -> random float input X[i,j] in [0,1] -> y = XOR of (X>0.5)*1
# TODO: learn to plot functions
# try to learn n-bit XOR
# create all tuples:


default_params = {
      "alpha" : 0.01,           # learning rate alpha - also triggers update of W,b in layers
      "alpha_min" : 0.01,       # when < alpha triggers adaptive learn rate
      "verbose" : 0,            # levels of verbsity 0-3
      "iterations" : 3000,      # how many iterations of grdient descent
      "epsilon" : 0.01,         # min/max element size for initialization of W
      "activations" : [act.TanH,act.Sigmoid],    # activation function either 2 or one for each layer
      "topology" : []           # array of layer sizes
      # "X": [[]]               # training input vectors
      # "Y": [[]]               # training output vectors
      # "X_t" : [[]]            # test input vectors
      # "Y_t" : [[]]            # test output vectors
    }


def adapt_lr(alpha_max, alpha_min, i_max, i_curr):
    b = - i_max / (1 - alpha_max/alpha_min)
    return alpha_max * b / (i_curr + b);


def init_soigt():
    np.random.seed(1)

    threshold = 20
    offset = 10
    size = 5
    m = 1000

    parameters = {
      "alpha" : 1,
      "alpha_min": 0.005,
      "verbose" : 0,
      "iterations" : 25000,
      "epsilon" : 0.5,
      "topology" : [size],
      "activations" : [act.TanH,act.Sigmoid],
    }

    width = size - 1;

    while width > 1:
        # for i in range(0,2):
        parameters["topology"].append(width)
        if width % 2 == 1:
            width -= 1
        else:
            width = int(width/2)
    parameters["topology"].append(1)

    print("topology:" + str(parameters["topology"]))

    parameters["X"] = x = np.random.randn(size,m) * threshold + offset
    parameters["Y"] = ((np.sum(x,axis=0,keepdims=True) / size) > offset) * 1
    parameters["X_t"] = x = np.random.randn(size, int(m/5)) * threshold + offset
    parameters["Y_t"] = ((np.sum(x, axis=0, keepdims=True) / size) > offset) * 1

    if parameters["verbose"] > 1:
        print("X:  " + print_matrix(parameters["X"],4))
        print("Y:  " + print_matrix(parameters["Y"], 4))

    return parameters


def init_xor():
    np.random.seed(1)

    size = 4

    parameters = {
      "alpha" : 0.1,
      "verbose" : 0,
      "iterations" : 10000,
      "epsilon" : 0.5,
      "topology" : [],
      "activations" : [act.TanH,act.Sigmoid],
    }

    width = size
    while width > 1:
        for i in range(0,2):
            parameters["topology"].append(width)
        if width % 2 == 1:
            width -= 1
        else:
            width = int(width/2)
    parameters["topology"].append(1)

    print("topology:" + str(parameters["topology"]))

    x = np.zeros((size,size * size))
    y = np.zeros((1,size*size))
    for i in range(0,size*size):
        for j in range(0,size):
            x[j,i] = (i & (1 << j)) > 0
            y[0,i] = x[0,i]
            for k in range(1,size):
                y[0,i] = y[0,i] != x[k,i]

    parameters["X"] = x
    parameters["Y"] = y
    return parameters


def learn(parameters):
    network = cn.create_network(parameters["topology"],parameters["activations"],parameters["epsilon"])
    print("network:\n" + str(network) + "\n")

    x = parameters["X"]
    y = parameters["Y"]
    verbose = parameters["verbose"]
    iterations = parameters["iterations"]

    if parameters["verbose"] > 0:
        print("X:    " + print_matrix(parameters["X"],6))
        print("Y:    " + print_matrix(parameters["Y"],6))

    params = {"Y":y, "backprop":True, "alpha":parameters["alpha"] }

    if verbose > 2:
        params["verbose"] = True

    for i in range(0,iterations):
        if ("alpha_min" in parameters) & (i > 0) & ((i % 100) == 0):
            if parameters["alpha_min"] < parameters["alpha"]:
                params["alpha"] = adapt_lr(parameters["alpha"],parameters["alpha_min"],iterations,i)

        if (verbose > 1) & (((i % (iterations / 10)) == 0)):
            params["verbose"] = True
            print("iteration: " + str(i))
        res = network.process(x, params)
        if (verbose < 3) & ("verbose" in params):
            del params["verbose"]
        if verbose > 1:
            print("Y_hat:" + print_matrix(res["Y_hat"], 6) + "\n")
            print("Y    :" + print_matrix(params["Y"], 6) + "\n")
        if (verbose > 2) | (((i % (iterations / 10)) == 0) & ("cost" in res)):
            print("{:5d}".format(i) + " - cost:" + str(res["cost"]))
            if verbose > 1:
                print("***********************************************")

    if verbose >= 1:
        params["verbose"] = True
    res = network.process(x, params)
    if "cost" in res:
        print("last -  cost:" + str(res["cost"]))

    res = network.process(x,{})
    err = res["Y_hat"] - parameters["Y"]
    acc = (1 - np.squeeze(np.dot(err,err.T))/parameters["Y"].shape[1]) * 100
    print("training accuracy:" + str(acc) + "%")

    res = network.process(parameters["X_t"],{})
    err = res["Y_hat"] - parameters["Y_t"]
    acc = (1 - np.squeeze(np.dot(err,err.T))/parameters["Y_t"].shape[1]) * 100
    print("test accuracy:" + str(acc) + "%")

    return network


# learn(init_xor())
learn(init_soigt())

# alpha_max = 0.1
# alpha_min = 0.01
# i_max = 10000
# i_cur = 0


# print("alpha max:" + "{:6.2f}".format(alpha_max) + " min:" + "{:6.2f}".format(alpha_min) + " i max:" + "{:6d}".format(i_max) +
#       " curr:" + "{:6d}".format(i_cur) + " alpha:" + "{:8.4f}".format(adapt_lr(alpha_max,alpha_min,i_max,i_cur)))

# for i in range(0,21):
#    i_cur = int(i_max * i / 20
#    print("alpha max:" + "{:6.2f}".format(alpha_max) + " min:" + "{:6.2f}".format(alpha_min) + " i max:" + "{:6d}".format(i_max) +
#    " curr:" + "{:6d}".format(i_cur) + " alpha:" + "{:8.4f}".format(adapt_lr(alpha_max,alpha_min,i_max,i_cur)))
