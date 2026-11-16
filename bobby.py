#Bobby's brain: one neuron
import numpy as np
import math

B_input = np.array([0.1,0.4,0.2])
B_weights = np.array([0.1,0.2,0.3])
B_bias = 0.1

#simple sigmoid
def sigmoid(x):
  return 1/(1+ np.exp(-x))

B_total = np.dot(B_input, B_weights) + B_bias
B_output = sigmoid(B_total)
print("bobby's brain: ", B_output)
#show amount of arrays
print("Bobby's total neurons:", B_input.size, "Bobby is very stupid!") #I'm not sure if it means that though
