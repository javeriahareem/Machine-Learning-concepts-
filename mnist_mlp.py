import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to [0, 1]
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create the model
model = Sequential()
# 1st hidden layer (we also need to tell the input dimension)
# Hidden layer with 5 neurons
model.add(Dense(5, input_dim=784, activation='sigmoid'))
## 2nd hidden layer - YOU MAY TEST THIS
#model.add(Dense(10, activation='sigmoid'))
# Output layer with 10 neurons (one for each class)
model.add(Dense(10, activation='softmax'))
#model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, activation='tanh'))

opt = keras.optimizers.SGD(learning_rate=0.1)
#model.compile(optimizer=opt, loss='mse', metrics=['mse'])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Opetus - epokkeja 1 tai 100
#history = model.fit(x_tr, y_tr_2, epochs=100, verbose=0)
# Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=2)

# Plot the training loss curve
plt.plot(history.history['loss'])
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Evaluate the model on the training data
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
print(f'Classification accuracy (training data): {train_acc * 100:.2f}%')

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Classification accuracy (test data): {test_acc * 100:.2f}%')