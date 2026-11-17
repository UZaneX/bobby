#Bobby's brain: six neuron! bobby getting smort!
import numpy as np
import math

np.random.seed(0)

B_c = np.array([0.1,0.4,0.2]) # <- where it starts 
B_input = B_c[np.newaxis, :] # <- The input where Bobby learns, takes the data, copy of B_c
B_input.shape # ".shape" <- makes them have X AND Y; basically 2D
B_weights = np.random.randn(3,6)#<- weights, not sure what they are for.
B_Bias = 0.1

Learn = np.array([1,0.4,0.2,0.45,0.54,0.26]) #<- what i will learn


#sigmoid
def sigmoid(x):
    return 1/ (1 + np.exp(-x))
total = np.dot(B_input,B_weights) + B_Bias
B_output = sigmoid(total)
def get_error():
    total = np.dot(B_input,B_weights) + B_Bias
    B_output = sigmoid(total)
    return np.sum(np.abs(B_output - Learn))

old_error = get_error()
get_error()
new_error = get_error()



print("Bobby's Brain power:", B_output)
print("total Nuerons: ", B_output.size)
print("the old error: ", old_error, "the new error: ", new_error)
print("sum: ",old_error+new_error)
