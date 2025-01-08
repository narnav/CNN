import numpy as np

# Sigmoid function implementation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example usage
x = np.array([-10, -1, 0, 1, 10])
y = sigmoid(x)

print("Input: ", x)
print("Sigmoid Output: ", y)