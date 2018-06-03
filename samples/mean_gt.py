import numpy as np
import NNToolkit.activation as act
from NNToolkit.util import print_matrix
from NNToolkit.manage import learn
from NNToolkit.manage import create
from NNToolkit.manage import evaluate
from NNToolkit.util import save_params
from NNToolkit.util import read_params


def init_mean_gt(m = 1000, it = 2000):
    np.random.seed(1)

    threshold = 20
    offset = 10
    size = 10

    parameters = {"alpha": 0.01,
                  #"alpha_min": 0.005,
                  "verbose": 0,
                  "iterations": it,
                  "lambda": 8,
                  "graph" : True,
                  "topology": [size,int(size/2),1],
                  "activations": [act.ReLU,act.Sigmoid]  # [act.TanH, act.Sigmoid]
                  }

#    width = size - 1

#    while width > 1:
        # for i in range(0,2):
#        parameters["topology"].append(width)
#        if width % 2 == 1:
#            width -= 1
#        else:
#            width = int(width/2)
#    parameters["topology"].append(1)

    print("topology:" + str(parameters["topology"]))

    parameters["X"] = x = np.random.randn(size,m)
    parameters["Y"] = np.int64((np.sum(x,axis=0,keepdims=True) / size) > 0)
    parameters["X_t"] = x = np.random.randn(size, int(m/5))
    parameters["Y_t"] = np.int64((np.sum(x, axis=0, keepdims=True) / size) > 0)

    if parameters["verbose"] > 1:
        print("X:  " + print_matrix(parameters["X"],4))
        print("Y:  " + print_matrix(parameters["Y"], 4))

    # save_json_params(parameters,"testCases/sum_of_int_gt.pkl")
    return parameters

def train_gt_mean(n = 1000):
    params = init_mean_gt(n)
    network = learn(params)
    network.get_weights(params)
    save_params(params, "../testCases/mean_gt_" + str(n) + ".json.gz")


def restore_gt_mean(n = 1000):
    params = read_params("../testCases/mean_gt_" + str(n) + ".json.gz")
    network = create(params)
    y_hat, acc = evaluate(network, params["X"], params["Y"])
    print("training accuracy:" + str(acc*100) + "%")

    y_hat, acc = evaluate(network,params["X_t"],params["Y_t"])
    print("test accuracy:    " + str(acc*100) + "%")


train_gt_mean(2000)

# restore_gt_mean()



