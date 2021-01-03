import numpy as np

# inputs = [1, 2, 3, 2.5]   # Shape is 4x1

# actually make the input batches

inputs = [[1, 2, 3, 2,5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]   # Shape is 3x4
 
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]   # Shape is 3x4

biases = [2, 3, 0.5]


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
# outputs = np.dot (weights, inputs) + biases
# print (outputs)




 
