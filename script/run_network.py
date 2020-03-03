import network
from load_mnist import load_data

# net = network.Network([784, 196, 49, 10])
net = network.Network([784, 30, 10])

train_data, test_data = load_data()

net.sgd(train_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
