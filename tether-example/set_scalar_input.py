import numpy as np


def get_input_activations(input_shapes):
    return [np.full(shape, np.float32(0.8)) for shape in input_shapes]

