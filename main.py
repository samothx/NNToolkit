import numpy as np
from NNLayer import layer,terminal
import NNLayer.create_network as cn
import NNLayer.activation as act

# try to learn n-bit XOR
# create all tuples:
size = 2

x = np.zeros((size,size * size))
y = np.zeros((1,size*size))
for i in range(0,size*size):
    for j in range(0,size):
        x[j,i] = (i & (1 << j)) > 0
        y[0,i] = x[0,i]
        for k in range(1,size):
            y[0,i] = y[0,i] != x[k,i]

print("X:\n" + str(x) + "\n")
print("X:\n" + str(y) + "\n")

sizes = [size,size,1]
layer0 = cn.create_network(sizes,[act.TanH,act.Sigmoid])
print("layer:\n" + str(layer0))
params = {"Y":y, "backprop":True, "update":True, "alpha":0.01 }
res = layer0.process(x,params)
print("Y_hat:" + str(res["Y_hat"]) + "\n")
print("Y    :" + str(params["Y"]) + "\n")
if "cost" in res:
    print("cost:" + str(res["cost"]))

for i in range(0,10000):
    res = layer0.process(x, params)
    if (i > 0) & ((i % 100) == 0) & ("cost" in res):
        print(str(i) + " - cost:" + str(res["cost"]))





