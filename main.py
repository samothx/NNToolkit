import numpy as np
from NNLayer import layer,terminal
import NNLayer.create_network as cn
import NNLayer.activation as act

from NNLayer.util import print_matrix

# try to learn n-bit XOR
# create all tuples:

size = 2
verbose = False
iterations = 100

x = np.zeros((size,size * size))
y = np.zeros((1,size*size))
for i in range(0,size*size):
    for j in range(0,size):
        x[j,i] = (i & (1 << j)) > 0
        y[0,i] = x[0,i]
        for k in range(1,size):
            y[0,i] = y[0,i] != x[k,i]

network = cn.create_network([size,2*size,2*size,1],[act.TanH,act.Sigmoid])
print("network:\n" + str(network) + "\n")

print("X:  " + print_matrix(x,6))
print("Y:  " + print_matrix(y,6))


params = {"Y":y, "backprop":True, "update":True, "alpha":0.05 }
if verbose:
    params["verbose"] = True

for i in range(0,iterations):
    res = network.process(x, params)
    if verbose:
        print("Y_hat:" + print_matrix(res["Y_hat"], 6) + "\n")
        print("Y    :" + print_matrix(params["Y"], 6) + "\n")
    if verbose | (((i % (iterations / 10)) == 0) & ("cost" in res)):
        print("{:5d}".format(i) + " - cost:" + str(res["cost"]))
    if verbose:
        print("***********************************************")

if "cost" in res:
    print("last -  cost:" + str(res["cost"]))





