import network2
from load_mnist import load_data

# net = network2.Network([784, 196, 49, 10])
net = network2.Network([784, 30, 10])

train_data, validation_data, test_data = load_data()

net.sgd(train_data,
        epochs=30,
        mini_batch_size=10,
        eta=0.5,
        lmbda=1.0E-4,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True,
        monitor_weight_stats=True)
