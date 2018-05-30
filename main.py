import numpy as np
import NNLayer.create_network as cn
import NNLayer.activation as act

from NNLayer.util import print_matrix

# try to learn n-bit XOR
# create all tuples:

default_params = {
      "alpha" : 0.01,
      "verbose" : False,
      "iterations" : 3000,
      "epsilon" : 0.01,
      "activations" : [act.TanH,act.Sigmoid]
    }

def init_xor():
    np.random.seed(1)

    size = 4

    parameters = {
      "alpha" : 0.1,
      "verbose" : False,
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

    print("X:    " + print_matrix(parameters["X"],6))
    print("Y:    " + print_matrix(parameters["Y"],6))

    params = {"Y":y, "backprop":True, "alpha":parameters["alpha"] }
    if verbose:
        params["verbose"] = True


    for i in range(0,iterations):
        if (verbose == False) & (((i % (iterations / 10)) == 0)):
            params["verbose"] = True
            print("iteration: " + str(i))
        res = network.process(x, params)
        if (verbose == False) & ("verbose" in params):
            del params["verbose"]
        if verbose:
            print("Y_hat:" + print_matrix(res["Y_hat"], 6) + "\n")
            print("Y    :" + print_matrix(params["Y"], 6) + "\n")
        if verbose | (((i % (iterations / 10)) == 0) & ("cost" in res)):
            print("{:5d}".format(i) + " - cost:" + str(res["cost"]))
            print("***********************************************")

    params["verbose"] = True
    res = network.process(x, params)
    if "cost" in res:
        print("last -  cost:" + str(res["cost"]))


learn(init_xor())
