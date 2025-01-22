import numpy as np
# ReLU stands for Rectified Linear Unit
# ReLU function
def relu(x):
    return np.maximum(0, x)

# Example usage
x = np.array([-10, -1, 0, 1, 10])  # Input values
y = relu(x)

print("Input: ", x)
print("ReLU Output: ", y)
