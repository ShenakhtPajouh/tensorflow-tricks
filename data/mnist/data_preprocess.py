import numpy as np


def get_dataset(path):
    data = np.loadtxt(path, delimiter=',')
    labels = data[:, 0]
    labels = labels.astype(int)
    images = data[:, 1:]
    images = images.reshape([images.shape[0], 28, 28])
    return images, labels

if __name__ == "__main__":
    train_images, train_labels = get_dataset("mnist_train.csv")
    test_images, test_labels = get_dataset("mnist_test.csv")
    np.savez('mnist.npz', test_images=test_images, test_labels=test_labels, train_images=train_images,
             train_labels=train_labels)

