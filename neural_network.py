import numpy as np

# inputs = [1, 2, 3, 2.5]   # Shape is 4x1

# actually make the input batches

# inputs = [[1, 2, 3, 2.5], 
#           [2.0, 5.0, -1.0, 2.0], 
#           [-1.5, 2.7, 3.3, -0.8]]        # Shape is (3x4)

 
# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]   # Shape is (3x4)
# biases = [2, 3, 0.5]                     # Shape is (3, )

# weights2 = [[0.1, -0.14, 0.5],
#            [-0.5, 0.12, -0.33],
#            [-0.44, 0.73, -0.13]]         # Shape is (3x4)
# biases2 = [-1, 2, -0.5]                  # Shape is (3, )


"""np.random.seed (0)
X = [[1, 2, 3, 2.5], 
     [2.0, 5.0, -1.0, 2.0], 
     [-1.5, 2.7, 3.3, -0.8]]        # Shape is (3x4)
"""

# DATASET
def create_data (points, classes):
    X = np.zeros ((points * classes, 2))
    y = np.zeros (points * classes, dtype= "uint8")
    for class_number in range (classes):
        ix = range (points * class_number, points * (class_number + 1))
        r = np.linspace (0.0, 1, points) # radius
        t = np.linspace (class_number * 4, (class_number + 1) * 4, points) + np.random.randn (points) * 0.2
        X[ix] = np.c_[r * np.sin (t * 2.5), r * np.cos (t * 2.5)]
        y[ix] = class_number
    return X, y

import matplotlib.pyplot as plt

print ("here")
X, y = create_data (100, 3)

plt.scatter (X[:, 0], X[:, 1])
plt.show ()

plt.scatter (X[:, 0], X[:, 1], c = y, cmap= 'brg')
plt.show ()



import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init ()
X, y = spiral_data (100, 3)                 # 100 feature sets of 3 classes
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """
        Constructor function for the class

        Params:
                n_inputs: size of the input layer coming in
                n_neurons: how many neurons should this layer have
        Outputs:
                No outputs. Just set the weights and biases on a layer based on the params
        """
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)                     # Shape of weights (n_inputs, n_neurons)
        self.biases = np.zeros ((1, n_neurons))                                        # Shape of baises (1, n_neurons)

    def forward (self, inputs):
        """
        Set the weights and biases for the next layer
        Params:
                inputs: the inputs of the previous layer
        """
        self.output = np.dot (inputs, self.weights) + self.biases

    
class Activation_ReLU:
    def forward (self, inputs):
        self.output = np.maximum (0, inputs)

layer1 = Layer_Dense (2,5)
activation1= Activation_ReLU ()

layer2 = Layer_Dense (5,2)                                       # layer2 first param has to match layer1 last param

layer1.forward (X)
# print (layer1.output)
activation1.forward (layer1.output)
print (activation1.output)
# print (layer1.output)
# layer2.forward (layer1.output)
# print (layer2.output)

# SHAPES 
"""
Array                                                Shape                                   Type (in Numpy , in Mathematics)

l = [1,2,3,4]                                        (4, )                                          1D aray,       Vector

list of list (lol) = [[1, 5, 6, 2],
                      [3, 2, 1, 3]]                  (2, 4)                                         2D array,       Matrix

outer list has 2 lists so shape is 2
and inner list has 4 elements so 4



lolol = [[[1, 5, 6, 2],
          [3, 2, 1, 3]],
         [[5, 2, 1, 2],
          [6, 4, 8, 4]],
         [[2, 8, 5, 3],         
          [1, 1, 9, 4]]]                              (3, 2, 4)                                       3D array,     Matrix (I suppose)

outer list has 3 lists
which in turn has 2 lists and 
which in turn has 4 elements      
"""      

# TENSOR IS AN OBJECT THAT CAN BE REPRESENTED AS AN ARRAY




# zip combines two lists together at index level so it will return a list
# ex. zip (weights, biases) => 0th index will be [weights[0]biases[0]]

# you can say this the long way to do this output_neuron = activation * weights + bias
# layer_outputs = []
# for neuron_weights, neuron_bias in zip (weights, biases):
#     neuron_output = 0;
#     for n_input, weight in zip (inputs, neuron_weights):
#         neuron_output += n_input*weight
#     neuron_output += neuron_bias
#     layer_outputs.append (neuron_output)
# print (layer_outputs)


# shorter way to do output_neuron = activation * weights + bias


# layer1_outputs = np.dot (inputs, np.array (weights).T) + biases
# layer2_outputs = np.dot (layer1_outputs, np.array (weights2).T) + biases2

# print (layer2_outputs)