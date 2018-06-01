import numpy as np
import NNToolkit.activation as act
from NNToolkit.util import divide2sets
from NNToolkit.util import print_matrix
from NNToolkit.util import save_params
import scipy.io
from NNToolkit.manage import learn

def init_hand_writing():
    matrix = scipy.io.loadmat("../data/mlnn.mat")
    assert isinstance(matrix["X"],np.ndarray)
    assert isinstance(matrix["y"], np.ndarray)

    x_raw = matrix["X"]
    y_raw = matrix["y"]

    #print("X_raw:" + print_matrix(x_raw,2))
    #print("Y_raw:" + print_matrix(y_raw, 2))
    #print("mean X_raw:" + str(np.sum(x_raw) / x_raw.shape[0] / x_raw.shape[1]))
    #print("max X_raw:" + str(np.max(x_raw)) + " min X_raw:" + str(np.min(x_raw)))

    m = x_raw.shape[0]
    n_0 = x_raw.shape[1]
    n_l = 10

    y_class = np.zeros((m,n_l))
    for i in range(0,n_l):
        y_class[:,i:i+1] = (y_raw == (i + 1))



    res = divide2sets(x_raw,y_class,0,0.01,True,True)

    print("shape: n0:" + str(n_0) + " nL:" + str(n_l) + " m:" + str(m))

    parameters = {"alpha": 0.5,
                  "alpha_min": 0.05,
                  "verbose": 0,
                  "iterations": 1000,
                  "epsilon": 0.01,
                  "topology": [n_0,int(3 * n_0 / 4),int(n_0/2), n_l],
                  "activations": [act.ReLU,act.Sigmoid],  # [act.TanH, act.Sigmoid]
                  "X" : res["X_train"],
                  "Y" : res["Y_train"]
                  }

    #print("X:" + print_matrix(parameters["X"],2))
    #print("Y:" + print_matrix(parameters["Y"], 2))

    if "X_test" in res:
        parameters["X_t"] = res["X_test"]
        parameters["Y_t"] = res["Y_test"]

    print("sum y_train:" + str(np.sum(res["Y_train"])))
    print("sum y:" + str(np.sum(y_raw > 0)))
    return parameters


params = init_hand_writing()
network = learn(params)
network.get_weights(params)
save_params(params,"../testCases/handWr_1000_t_92.json.gz")
