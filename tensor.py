import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import os



# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]
x_train = x_train[..., tf.newaxis]  # Add channel dimension
x_test = x_test[..., tf.newaxis]

if os.path.exists('my_cnn_model.h5'):
    print("Model saved successfully and the file exists.")
    sample_image = x_test[0:1]  # Select the first test image (add batch dimension)
    model = load_model('my_cnn_model.h5')
    predictions = model.predict(sample_image)

    # Output probabilities
    print("Predicted probabilities:", predictions)

    # Output the predicted class
    predicted_class = np.argmax(predictions)
    print(f"Predicted Class: {predicted_class}")

    # Ground truth
    print(f"Actual Class: {y_test[0]}")
    exit()



# Assuming x_train is already loaded (e.g., from MNIST dataset)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Display the first image in x_train
# plt.imshow(x_train[0], cmap='gray')  # Use 'gray' colormap for grayscale images
# plt.title(f"Label: {y_train[0]}")   # Display the corresponding label
# plt.axis('off')                     # Hide axes
# plt.show()

# print(x_train[0])
# exit()


# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Save the model in SavedModel in HDF5 Format (.h5):
model.save('my_cnn_model.h5')
print("Model saved in TensorFlow SavedModel format.")

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# Validation Accuracy is a metric that indicates how well a machine learning model performs on a validation dataset,
#  which is a subset of data not used during training.
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
