import numpy as np
from NNToolkit.util import print_matrix

import math
import matplotlib.pyplot as plt

# TODO: input normalization
# TODO: learn to plot functions


# x = np.random.randn(40,40)
# print("x:",print_matrix(x,3))

x = np.arange(0,2 * math.pi,0.1)
print("shape x:" + str(x.shape))
ys = np.sin(x)
yc = np.cos(x)
plt.plot(x,ys)
plt.plot(x,yc)
plt.ylabel('sin x')
plt.show()

# mpl.plot(x,y)


# mpl.plot()

