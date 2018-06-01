import numpy as np
import NNToolkit.activation as act
from NNToolkit.util import read_matlab_file
from NNToolkit.manage import learn

def init_hand_writing():
    matrix = read_matlab_file("../data/mlnn.mat")
    # assert isinstance(matrix, np.ndarray)
    assert isinstance(matrix["X"],np.ndarray)
    assert isinstance(matrix["y"], np.ndarray)
    x = matrix["X"].T
    y = matrix["y"].T

    n_0 = x.shape[0]
    n_l = 10
    m = x.shape[1]

    print("shape: n0" + str(n_0) + " nL:" + str(n_l) + " m:" + str(m))

    y_dig = np.zeros((n_l,m))

    for i in range(1,n_l + 1):
        y_dig[i-1, :] = (y == i)

    parameters = {"alpha": 0.03,
                  "alpha_min": 0.005,
                  "verbose": 0,
                  "iterations": 3000,
                  "epsilon": 0.01,
                  "topology": [n_0,int(3 * n_0 / 4),int(n_0/2), n_l],
                  "activations": [act.ReLU,act.Sigmoid],  # [act.TanH, act.Sigmoid]
                  "X" : x,
                  "Y" : y_dig
                  }

    print("sum y_dig:" + str(np.sum(y_dig)))
    print("sum y:" + str(np.sum(y > 0)))
    return parameters


params = init_hand_writing()
network = learn(params)




