# cnn implements with numpy

import numpy as np

# Activation function: ReLU
def relu(x):
    return np.maximum(0, x)

# Activation function: Softmax
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
    return exps / np.sum(exps, axis=1, keepdims=True)

# 2D Convolution
def conv2d(input, filters, stride=1, padding=0):
    n_filters, filter_height, filter_width = filters.shape
    input_height, input_width = input.shape
    output_height = (input_height - filter_height + 2 * padding) // stride + 1
    output_width = (input_width - filter_width + 2 * padding) // stride + 1

    if padding > 0:
        padded_input = np.pad(input, ((padding, padding), (padding, padding)), mode='constant')
    else:
        padded_input = input

    output = np.zeros((n_filters, output_height, output_width))
    for f in range(n_filters):
        for i in range(0, output_height):
            for j in range(0, output_width):
                region = padded_input[
                    i * stride:i * stride + filter_height,
                    j * stride:j * stride + filter_width
                ]
                output[f, i, j] = np.sum(region * filters[f])  # Convolution operation
    return output

# Max Pooling
def max_pooling(input, pool_size, stride):
    n_channels, input_height, input_width = input.shape
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1

    output = np.zeros((n_channels, output_height, output_width))
    for c in range(n_channels):
        for i in range(0, output_height):
            for j in range(0, output_width):
                region = input[
                    c,
                    i * stride:i * stride + pool_size,
                    j * stride:j * stride + pool_size
                ]
                output[c, i, j] = np.max(region)  # Max pooling
    return output

# Fully Connected Layer
def dense(input, weights, biases):
    return np.dot(input, weights) + biases

# CNN Forward Pass
def cnn_forward_pass(input_image, filters, pool_size, stride, fc_weights, fc_biases):
    # Step 1: Convolution
    conv_output = conv2d(input_image, filters)
    conv_output = relu(conv_output)  # Apply ReLU activation

    # Step 2: Max Pooling
    pooled_output = max_pooling(conv_output, pool_size=pool_size, stride=stride)

    # Step 3: Flatten
    flattened_output = pooled_output.flatten()

    # Step 4: Fully Connected Layer
    fc_output = dense(flattened_output, fc_weights, fc_biases)
    fc_output = softmax(fc_output)  # Apply softmax for classification

    return fc_output

# Example Usage
if __name__ == "__main__":
    # Input image: 6x6 grayscale
    input_image = np.array([
        [1, 2, 3, 0, 1, 2],
        [4, 5, 6, 1, 0, 3],
        [7, 8, 9, 2, 1, 4],
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 0, 1],
        [2, 3, 4, 5, 6, 7]
    ])

    # Filters (2 filters, 3x3 size)
    filters = np.array([
        [[1, 0, -1],
         [1, 0, -1],
         [1, 0, -1]],
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]]
    ])

    # Fully connected layer weights and biases
    fc_weights = np.random.randn(8, 10)  # Assuming 8 inputs and 10 classes
    fc_biases = np.random.randn(10)

    # Forward pass
    output = cnn_forward_pass(
        input_image=input_image,
        filters=filters,
        pool_size=2,
        stride=2,
        fc_weights=fc_weights,
        fc_biases=fc_biases
    )

    print("Output probabilities:", output)
