import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score
import numpy.linalg as la

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

# Compute the mean and full covariance matrix for each class
def compute_mean_covariance(x_train, y_train, num_classes=10):
    means = np.zeros((num_classes, x_train.shape[1]))
    covariances = np.zeros((num_classes, x_train.shape[1], x_train.shape[1]))
    
    for k in range(num_classes):
        class_data = x_train[y_train == k]
        means[k, :] = np.mean(class_data, axis=0)
        covariances[k, :, :] = np.cov(class_data, rowvar=False)  # Full covariance matrix
    
    # Add a small value to the diagonal of each covariance matrix to avoid singular matrices
    covariances += np.eye(x_train.shape[1]) * 0.001
    return means, covariances

# Predict using the multivariate Gaussian model for each class
def predict_multivariate(x_test, means, covariances, num_classes=10):
    log_likelihoods = np.zeros((x_test.shape[0], num_classes))
    
    for k in range(num_classes):
        # Compute logpdf of the multivariate normal for each test sample
        log_likelihoods[:, k] = multivariate_normal.logpdf(x_test, mean=means[k], cov=covariances[k])
    
    return np.argmax(log_likelihoods, axis=1)

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python mnist_full_bayes.py <original|fashion>")
        sys.exit(1)

    dataset_version = sys.argv[1]

    # Load dataset (original MNIST or Fashion MNIST)
    (x_train, y_train), (x_test, y_test) = load_data(dataset_version)

    # Optionally add noise to the training data
    x_train_noisy = add_noise(x_train, scale=0.1)  # Test with different noise scales

    # Compute the mean and covariance matrices for each class
    means, covariances = compute_mean_covariance(x_train_noisy, y_train)

    # Check if the covariance matrices are full rank
    for k in range(10):
        rank = la.matrix_rank(covariances[k])
        print(f"Class {k} covariance matrix rank: {rank}")

    # Classify the test data using multivariate normal
    y_pred = predict_multivariate(x_test, means, covariances)

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
