"""
mnist_svm
~~~~~~~~~
Support Vector Machine
A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""
from load_mnist import load_data
from sklearn import svm
import numpy as np


def svm_baseline():
    train_data, test_data = load_data()
    train_features = [e[0].reshape(784, ) for e in train_data]
    train_labels = [np.argmax(e[1].reshape(10, )) for e in train_data]

    # train
    clf = svm.SVC()
    print("DEBUG 01 -> before fit")
    clf.fit(train_features, train_labels)
    print("DEBUG 02 -> after fit")
    # test
    test_features = [e[0].reshape(784, ) for e in test_data]
    test_labels = [np.argmax(e[1].reshape(10, )) for e in test_data]

    predictions = [a for a in clf.predict(test_features)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_labels))
    print("Baseline classifier using an SVM.")
    print("%s of %s values correct." % (num_correct, len(test_labels)))


if __name__ == "__main__":
    svm_baseline()
