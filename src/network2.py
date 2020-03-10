"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights. The code is simple,
easily readable. It is not optimized, and omits many desirable features.
"""
import json
import random
import sys

import numpy as np
from typing import List, Tuple, TypeVar, cast

T = TypeVar("T", float, np.ndarray)


class QuadraticCost:

    @staticmethod
    def fn(a: np.ndarray, y: np.ndarray) -> float:
        """Return the cost associated with an output ``a`` and desired output ``y``.
        """
        return 0.5 * ((a - y) ** 2).sum()

    @staticmethod
    def delta(z: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost:

    @staticmethod
    def fn(a: np.ndarray, y: np.ndarray) -> float:
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return cast(float, np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))))

    @staticmethod
    def delta(z, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return a - y


# noinspection DuplicatedCode
class Network:

    def __init__(self, sizes: List[int], cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using ``self.default_weight_initializer``
        (see docstring for that method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases, self.weights = self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        incoming weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        return biases, weights

    def large_weight_initializer(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        return biases, weights

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self,
            training_data: List[Tuple[np.ndarray, np.ndarray]],
            epochs: int,
            mini_batch_size: int,
            eta: float,
            lmbda: float = 0.0,
            evaluation_data: List[Tuple[np.ndarray, np.ndarray]] = None,
            monitor_evaluation_cost: bool = False,
            monitor_evaluation_accuracy: bool = False,
            monitor_training_cost: bool = False,
            monitor_training_accuracy: bool = False,
            monitor_weight_stats: bool = False) -> Tuple[List[float], List[int], List[float], List[int]]:
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        n_data = None
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda)
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {:.2f}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.evaluate(training_data)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {:.2f}".format(cost))
            if monitor_evaluation_accuracy and evaluation_data:
                accuracy = self.evaluate(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_data))
            if monitor_weight_stats:
                self.weight_stats()
            print("--- --- ---")
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def weight_stats(self):
        w_min = np.concatenate([w.flatten() for w in self.weights]).min(initial=0)
        w_max = np.concatenate([w.flatten() for w in self.weights]).max(initial=0)
        w_mean = np.concatenate([w.flatten() for w in self.weights]).mean()
        w_std = np.concatenate([w.flatten() for w in self.weights]).std()
        print("Weights min|max|mean|std: {:.2f}|{:.2f}|{:.2f}|{:.2f}".format(w_min, w_max, w_mean, w_std))
        b_min = np.concatenate([w.flatten() for w in self.biases]).min(initial=0)
        b_max = np.concatenate([w.flatten() for w in self.biases]).max(initial=0)
        b_mean = np.concatenate([w.flatten() for w in self.biases]).mean()
        b_std = np.concatenate([w.flatten() for w in self.biases]).std()
        print("Biases min|max|mean|std: {:.2f}|{:.2f}|{:.2f}|{:.2f}".format(b_min, b_max, b_mean, b_std))

    def update_mini_batch(self,
                          mini_batch: List[Tuple[np.ndarray, np.ndarray]],
                          eta: float,
                          lmbda: float) -> None:
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * lmbda) * w - (eta / len(mini_batch)) * nw for w, nw in
                        zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # ll = 1 means the last layer of neurons, ll = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for ll in range(2, self.num_layers):
            z = zs[-ll]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-ll + 1].transpose(), delta) * sp
            nabla_b[-ll] = delta
            nabla_w[-ll] = np.dot(delta, activations[-ll - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data) -> int:
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(y[x]) for (x, y) in test_results)

    def total_cost(self, data, lmbda):
        """Return the total cost for the data set ``data``.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * lmbda * sum((w ** 2).sum() for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def sigmoid(z: T) -> T:
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: T) -> T:
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
