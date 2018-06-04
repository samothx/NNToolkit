from NNToolkit.manage import create
from NNToolkit.parameters.setup import SetupParams

# TODO: input normalization
# TODO: learn to plot functions


# x = np.random.randn(40,40)
# print("x:",print_matrix(x,3))

params = SetupParams()
params.topology = [10,5,1]

network = create(params)
print("network:" + str(network))



# mpl.plot(x,y)


# mpl.plot()

