import numpy as np
import NNToolkit.activation as act
from NNToolkit.util import print_matrix
from NNToolkit.manage import learn
from NNToolkit.manage import create
from NNToolkit.manage import evaluate
from NNToolkit.util import save_params
from NNToolkit.util import read_params
# from NNToolkit.manage import read_params

# TODO: input normalization
# TODO: learn to plot functions
# try to learn n-bit XOR
# create all tuples:


def init_mean_gt(m = 1000):
    np.random.seed(1)

    threshold = 20
    offset = 10
    size = 5

    parameters = {
      "alpha" : 1,
      "alpha_min": 0.005,
      "verbose" : 0,
      "iterations" : 25000,
      "epsilon" : 0.5,
      "topology" : [size],
      "activations" : [act.TanH,act.Sigmoid],
    }

    width = size - 1

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

    # save_json_params(parameters,"testCases/sum_of_int_gt.pkl")
    return parameters


def init_xor():

    # Fuzzy XOR -> random float input X[i,j] in rand[0,1] -> y = XOR of (X>0.5)
    # XOR is computed by extracting tmp = x[0] and then computing
    # tmp = tmp != x[i] for i in {1,n}

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

    np.random.seed(1)

    size = 4
    m = 100

    parameters = {
      "alpha" : 1,
      "alpha_min": 0.01,
      "verbose" : 0,
      "iterations" : 10000,
      "epsilon" : 0.7,
      "topology" : [size],
      "activations" : [act.Sigmoid],   # [act.TanH,act.Sigmoid],
    }

    width = size + 1
    while width > 1:
        for i in range(0,2):
            parameters["topology"].append(width)
        if width % 2 == 1:
            width -= 1
        else:
            width = int(width/2)
    parameters["topology"].append(1)

    print("topology:" + str(parameters["topology"]))


    x, y = rand_data(size,m)
    parameters["X"] = x
    parameters["Y"] = y

    x, y = rand_data(size, int(m/5))
    parameters["X_t"] = x
    parameters["Y_t"] = y

    # this did not work at all
    #x = np.zeros((size,size * size))
    #y = np.zeros((1,size*size))
    #for i in range(0,size*size):
    #    for j in range(0,size):
    #        x[j,i] = (i & (1 << j)) > 0
    #        y[0,i] = x[0,i]
    #        for k in range(1,size):
    #            y[0,i] = y[0,i] != x[k,i]

    return parameters


# test_adapt_alpha()
# learn(init_xor())

def train_gt_mean(n = 1000):
    params = init_mean_gt(n)
    network = learn(params)
    network.get_weights(params)
    save_params(params, "testCases/mean_gt_" + str(n) + ".json.gz")

def restore_gt_mean(n = 1000):
    params = read_params("testCases/mean_gt_" + str(n) + ".json.gz")
    network = create(params)
    y_hat, acc = evaluate(network, params["X"], params["Y"])
    print("training accuracy:" + str(acc*100) + "%")

    y_hat, acc = evaluate(network,params["X_t"],params["Y_t"])
    print("test accuracy:    " + str(acc*100) + "%")


# train_gt_mean()

restore_gt_mean()



