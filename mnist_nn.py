import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from random import random, randint  # Ensure randint is imported
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load data based on command-line argument
def load_data(version='original'):
    if version == 'fashion':
        mnist = tf.keras.datasets.fashion_mnist
    else:
        mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the size of training and test data
#print(f'x_train shape {x_train.shape}')
#print(f'y_train shape {y_train.shape}')
#print(f'x_test shape {x_test.shape}')
#print(f'y_test shape {y_test.shape}')

# Reshape images to vectors of size 28*28=784
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    # Normalize the data to the range [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    return (x_train, y_train), (x_test, y_test)


# Class to evaluate accuracy
class AccuracyEvaluator:
    def __init__(self):
        pass
    
    def acc(self, pred, gt):
        """
        Compute classification accuracy between predicted (pred) and ground truth (gt) labels.
        
        Args:
        pred: numpy array of predicted labels
        gt: numpy array of ground truth labels
        
        Returns:
        accuracy: classification accuracy as a float
        """
        assert len(pred) == len(gt), "Predictions and ground truth arrays must have the same length."
        correct = np.sum(pred == gt)
        accuracy = correct / len(gt)
        return accuracy

# Instantiate the accuracy evaluator
#evaluator = AccuracyEvaluator()

# Generate random predictions for all test samples (values between 0-9)
#random_preds = np.array([randint(0, 9) for _ in range(len(y_test))])

# Compute and print the accuracy with random predictions
#accuracy = evaluator.acc(random_preds, y_test)
#print(f"Accuracy with random predictions: {accuracy:.2%}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python mnist_nn.py <original|fashion>")
        sys.exit(1)

    dataset_version = sys.argv[1]
    
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_data(dataset_version)
    
    # Construct the 1-NN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    
    # Train the classifier
    knn.fit(x_train, y_train)
    
    # Test the classifier
    y_pred = knn.predict(x_test)
    
    # Evaluate accuracy
    evaluator = AccuracyEvaluator()
    accuracy = evaluator.acc(y_pred, y_test)
    
    # Print the classification accuracy
    print(f"Classification accuracy is {accuracy:.2%}")

    # Visualization
    for i in range(x_test.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1)
            plt.clf()
            # Reshape the vector back into a 28x28 image for visualization
            plt.imshow(x_test[i].reshape(28, 28), cmap='gray_r')
            plt.title(f"Image {i} label: {y_test[i]}, predicted: {y_pred[i]}")
            plt.pause(1)

if __name__ == "__main__":
    main()