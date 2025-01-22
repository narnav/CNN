import numpy as np
import os
from PIL import Image

# Helper functions for CNN components
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    n_samples = y_true.shape[0]
    log_probs = -np.log(y_pred[range(n_samples), y_true])
    return np.sum(log_probs) / n_samples

def cross_entropy_derivative(y_pred, y_true):
    n_samples = y_true.shape[0]
    grad = y_pred
    grad[range(n_samples), y_true] -= 1
    grad = grad / n_samples
    return grad

# Convolutional Layer
class ConvLayer:
    def __init__(self, input_channels, num_filters, filter_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size) * 0.1
        self.biases = np.zeros((num_filters, 1))

    def forward(self, x):
        self.input = x
        batch_size, input_channels, input_height, input_width = x.shape
        output_height = (input_height - self.filter_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.filter_size + 2 * self.padding) // self.stride + 1
        self.output = np.zeros((batch_size, self.num_filters, output_height, output_width))

        # Apply padding
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Convolve each filter
        for i in range(self.num_filters):
            for h in range(output_height):
                for w in range(output_width):
                    h_start = h * self.stride
                    h_end = h_start + self.filter_size
                    w_start = w * self.stride
                    w_end = w_start + self.filter_size

                    region = x[:, :, h_start:h_end, w_start:w_end]
                    self.output[:, i, h, w] = np.sum(region * self.filters[i], axis=(1, 2, 3)) + self.biases[i]
        return self.output

    def backward(self, d_out, learning_rate):
        batch_size, input_channels, input_height, input_width = self.input.shape
        _, _, output_height, output_width = d_out.shape
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input)

        # Apply padding to the input
        if self.padding > 0:
            self.input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            d_input = np.pad(d_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Backpropagation
        for i in range(self.num_filters):
            for h in range(output_height):
                for w in range(output_width):
                    h_start = h * self.stride
                    h_end = h_start + self.filter_size
                    w_start = w * self.stride
                    w_end = w_start + self.filter_size

                    region = self.input[:, :, h_start:h_end, w_start:w_end]
                    d_filters[i] += np.sum(region * (d_out[:, i, h, w])[:, None, None, None], axis=0)
                    d_input[:, :, h_start:h_end, w_start:w_end] += self.filters[i] * (d_out[:, i, h, w])[:, None, None, None]
            d_biases[i] += np.sum(d_out[:, i, :, :])

        # Remove padding from gradients if applied
        if self.padding > 0:
            d_input = d_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # Update filters and biases
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases
        return d_input

# Pooling Layer
class MaxPoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.input = x
        batch_size, input_channels, input_height, input_width = x.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        self.output = np.zeros((batch_size, input_channels, output_height, output_width))

        for h in range(output_height):
            for w in range(output_width):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                w_start = w * self.stride
                w_end = w_start + self.pool_size

                region = x[:, :, h_start:h_end, w_start:w_end]
                self.output[:, :, h, w] = np.max(region, axis=(2, 3))
        return self.output

    def backward(self, d_out):
        d_input = np.zeros_like(self.input)
        batch_size, input_channels, output_height, output_width = d_out.shape

        for h in range(output_height):
            for w in range(output_width):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                w_start = w * self.stride
                w_end = w_start + self.pool_size

                region = self.input[:, :, h_start:h_end, w_start:w_end]
                max_region = np.max(region, axis=(2, 3), keepdims=True)
                mask = (region == max_region)
                d_input[:, :, h_start:h_end, w_start:w_end] += mask * (d_out[:, :, h, w])[:, :, None, None]
        return d_input

# Fully Connected Layer
class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, d_out, learning_rate):
        d_weights = np.dot(self.input.T, d_out)
        d_biases = np.sum(d_out, axis=0, keepdims=True)
        d_input = np.dot(d_out, self.weights.T)

        # Update weights and biases
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_input

# Training Configuration
def train_model():
    np.random.seed(42)
    
    # Mock data (Tensor Dataset)
    num_samples = 100
    input_height, input_width = 28, 28
    num_classes = 10

    X_train = np.random.rand(num_samples, 1, input_height, input_width)  # Grayscale images
    y_train = np.random.randint(0, num_classes, size=num_samples)

    # Initialize Layers
    conv1 = ConvLayer(input_channels=1, num_filters=8, filter_size=3, stride=1, padding=1)
    pool1 = MaxPoolLayer(pool_size=2, stride=2)
    fc1 = FCLayer(input_size=14 * 14 * 8, output_size=128)
    fc2 = FCLayer(input_size=128, output_size=num_classes)

    # Training Loop
    num_epochs = 10
    learning_rate = 0.01
    batch_size = 10

    for epoch in range(num_epochs):
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward Pass
            out = conv1.forward(X_batch)
            out = relu(out)
            out = pool1.forward(out)
            out = out.reshape(out.shape[0], -1)  # Flatten
            out = fc1.forward(out)
            out = relu(out)
            out = fc2.forward(out)
            out = softmax(out)

            # Loss
            loss = cross_entropy_loss(out, y_batch)

            # Backward Pass
            d_out = cross_entropy_derivative(out, y_batch)
            d_out = fc2.backward(d_out, learning_rate)
            d_out = relu_derivative(d_out) * d_out
            d_out = fc1.backward(d_out, learning_rate)
            d_out = d_out.reshape(-1, 8, 14, 14)  # Reshape
            d_out = pool1.backward(d_out)
            d_out = relu_derivative(d_out) * d_out
            conv1.backward(d_out, learning_rate)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # Save Model
    np.save("conv1_filters.npy", conv1.filters)
    np.save("conv1_biases.npy", conv1.biases)
    np.save("fc1_weights.npy", fc1.weights)
    np.save("fc1_biases.npy", fc1.biases)
    np.save("fc2_weights.npy", fc2.weights)
    np.save("fc2_biases.npy", fc2.biases)
    print("Model saved!")

# Load Model and Predict
def load_model():
    conv1 = ConvLayer(input_channels=1, num_filters=8, filter_size=3, stride=1, padding=1)
    pool1 = MaxPoolLayer(pool_size=2, stride=2)
    fc1 = FCLayer(input_size=14 * 14 * 8, output_size=128)
    fc2 = FCLayer(input_size=128, output_size=10)

    # Load weights and biases
    conv1.filters = np.load("conv1_filters.npy")
    conv1.biases = np.load("conv1_biases.npy")
    fc1.weights = np.load("fc1_weights.npy")
    fc1.biases = np.load("fc1_biases.npy")
    fc2.weights = np.load("fc2_weights.npy")
    fc2.biases = np.load("fc2_biases.npy")

    return conv1, pool1, fc1, fc2

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array[np.newaxis, np.newaxis, :, :]  # Shape (1, 1, 28, 28)

def predict(conv1, pool1, fc1, fc2, image_array):
    out = conv1.forward(image_array)
    out = relu(out)
    out = pool1.forward(out)
    out = out.reshape(out.shape[0], -1)  # Flatten
    out = fc1.forward(out)
    out = relu(out)
    out = fc2.forward(out)
    out = softmax(out)
    return np.argmax(out, axis=1)

def run_inference_on_images():
    conv1, pool1, fc1, fc2 = load_model()

    media_dir = "media"
    if not os.path.exists(media_dir):
        print(f"Directory '{media_dir}' does not exist.")
        return

    for image_name in os.listdir(media_dir):
        image_path = os.path.join(media_dir, image_name)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_array = preprocess_image(image_path)
            prediction = predict(conv1, pool1, fc1, fc2, image_array)
            print(f"Image: {image_name}, Predicted Class: {prediction[0]}")

# Uncomment to train the model
train_model()

# Uncomment to run inference on images
# run_inference_on_images()
