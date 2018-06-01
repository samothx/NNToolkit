import numpy as np
import NNToolkit.activation as act
from NNToolkit.util import divide2sets
from NNToolkit.util import print_matrix
import scipy.io
from NNToolkit.manage import learn

def init_hand_writing():
    matrix = scipy.io.loadmat("../data/mlnn.mat")
    assert isinstance(matrix["X"],np.ndarray)
    assert isinstance(matrix["y"], np.ndarray)
    x_raw = matrix["X"]
    y_raw = matrix["y"]

    res = divide2sets(x_raw,y_raw,0.01,0.01,True,True)




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
                  "iterations": 100,
                  "epsilon": 0.01,
                  "topology": [n_0,int(3 * n_0 / 4),int(n_0/2), n_l],
                  "activations": [act.ReLU,act.Sigmoid],  # [act.TanH, act.Sigmoid]
                  "X" : x,
                  "Y" : y_dig
                  }

    print("sum y_dig:" + str(np.sum(y_dig)))
    print("sum y:" + str(np.sum(y > 0)))
    return parameters


# params = init_hand_writing()
# network = learn(params)

x = np.zeros((3,10))
val = 1
for i in range(0,x.shape[1]):
    for j in range(0, x.shape[0]):
        x[j,i] = val
        val += 1


y = np.zeros((2,x.shape[1]))
y[0,:] = np.sum(x,axis=0,keepdims=True)
y[1,:] = np.mean(x,axis=0,keepdims=True)

print("x:      " + print_matrix(x,8))
print("y:      " + print_matrix(y,8))

res = divide2sets(x.T,y.T,0.3,0.1,True,True)

if "X_train" in res:
    print("x_train:" + print_matrix(res["X_train"],8))
    print("y_train:" + print_matrix(res["Y_train"],8))

if "X_cv" in res:
    print("x_cv:   " + print_matrix(res["X_cv"],8))
    print("y_cv:   " + print_matrix(res["Y_cv"],8))

if "X_test" in res:
    print("x_test: " + print_matrix(res["X_test"],8))
    print("y_test: " + print_matrix(res["Y_test"],8))


