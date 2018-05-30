from NNLayer.terminal import Terminal
import NNLayer.activation as act
from NNLayer.layer import Layer


def create_network(layer_sizes, activations=(act.ReLU, act.Sigmoid), epsilon = 0.01, local_params=True):
    layers = len(layer_sizes)
    assert layers > 0

    if len(activations) >= layers:
        root = Layer(layer_sizes[0], activations[0], epsilon,local_params)
        for i in range(1, layers):
            root.add_layer(Layer(layer_sizes[i], activations[i],epsilon, local_params))
    else:
        first_act = activations[0]
        last_act = activations[1]
        if layers == 1:
            first_act = activations[1]
        root = Layer(layer_sizes[0], first_act(),epsilon, local_params)
        for i in range(1, layers - 1):
            root.add_layer(Layer(layer_sizes[i], first_act(),epsilon, local_params))
        root.add_layer(Layer(layer_sizes[layers - 1], last_act(), epsilon, local_params))

    root.add_layer(Terminal())
    return root
