# 253_PA2

The final_version.py file has the final code (adapted from the starter: neuralnet.py).

final_version.py can be run from the same directory that contains config.yaml, and it plots the required graphs for the parameters specified by the config file, and reports the test accuracy.

It also prints out numerical approximates of gradients, actual gradient values and their difference for 10 training examples, each of a different category, for the weight specified in the main function's call to gradient function. 
The weight can be specified in this call by passing the values of the layer, the input and output indices and whether or not it's a bias unit.
