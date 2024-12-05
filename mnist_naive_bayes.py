import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load MNIST or Fashion MNIST dataset based on the argument
def load_data(version='original'):
    if version == 'fashion':
        mnist = tf.keras.datasets.fashion_mnist
    else:
        mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize data to [0, 1] and reshape images into vectors
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0
    
    return (x_train, y_train), (x_test, y_test)

# Add Gaussian noise to the training data
def add_noise(x_train, scale=0.1):
    noise = np.random.normal(loc=0.0, scale=scale, size=x_train.shape)
    return x_train + noise

# Compute the mean and variance for each class
def compute_mean_variance(x_train, y_train, num_classes=10):
    means = np.zeros((num_classes, x_train.shape[1]))
    variances = np.zeros((num_classes, x_train.shape[1]))
    
    for k in range(num_classes):
        class_data = x_train[y_train == k]
        means[k, :] = np.mean(class_data, axis=0)
        variances[k, :] = np.var(class_data, axis=0)
    
    # Add a small value to variances to avoid division by zero
    variances += 0.001
    return means, variances

# Compute the log likelihood for a given class
def compute_log_likelihood(x, mean, variance):
    # Apply the log likelihood formula from (7)
    log_likelihood = -0.5 * np.sum(
        np.log(2 * np.pi * variance) + ((x - mean) ** 2) / variance, axis=1
    )
    return log_likelihood

# Classify each test sample based on the highest log likelihood
def predict(x_test, means, variances, num_classes=10):
    log_likelihoods = np.zeros((x_test.shape[0], num_classes))
    
    for k in range(num_classes):
        log_likelihoods[:, k] = compute_log_likelihood(x_test, means[k], variances[k])
    
    return np.argmax(log_likelihoods, axis=1)

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python mnist_naive_bayes.py <original|fashion>")
        sys.exit(1)

    dataset_version = sys.argv[1]

    # Load dataset (original MNIST or Fashion MNIST)
    (x_train, y_train), (x_test, y_test) = load_data(dataset_version)

    # Optionally add noise to the training data
    x_train_noisy = add_noise(x_train, scale=0.1)  # Test with different noise scales

    # Compute the mean and variance for each class
    means, variances = compute_mean_variance(x_train_noisy, y_train)

    # Classify the test data
    y_pred = predict(x_test, means, variances)

    # Compute and print the classification accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy is {accuracy:.2%}")

    # Visualize some test images and their predictions
    for i in range(10):
        plt.figure()
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"True Label: {y_test[i]}, Predicted: {y_pred[i]}")
        plt.show()

if __name__ == "__main__":
    main()
