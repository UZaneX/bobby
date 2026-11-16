#Bobby's brain: six neuron! bobby getting smort!
import numpy as np
import math

B_c = np.array([0.1,0.4,0.2]) # <- where it starts 
B_input = B_c[np.newaxis, :] # <- The input where Bobby learns, takes the data, copy of B_c
B_input.shape # ".shape" <- makes them have X AND Y; basically 2D

#weights
B_weights = np.random.randn(3,6) * 1
#bias
B_Bias = np.array([0.2])

#B2_i and B_input each have 3 Neurons.

#sigmoid
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

total = np.dot(B_input,B_weights) + B_Bias

B_output = sigmoid(total)

print("Bobby's Brain power:", B_output)

print("total Nuerons: ", B_output.size)
