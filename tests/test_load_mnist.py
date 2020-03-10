from load_mnist import load_data
import numpy as np


def test_load_data():
    data = load_data()
    assert len(data) == 3  # 3 data sets
    train_data = list(data[0])
    validation_data = list(data[1])
    test_data = list(data[2])

    assert len(train_data) == 50000  # 50,000 training images
    assert len(validation_data) == 10000  # 10,000 testing images
    assert len(test_data) == 10000  # 10,000 testing images

    for data_set in [train_data, validation_data, test_data]:
        assert len(data_set[0][0]) == 784  # feature is a vector of size 784, 28x28 pixels image flattened
        assert type(data_set[0][0]) == np.ndarray
        assert data_set[0][0].shape == (784, 1)

        assert len(data_set[0][1]) == 10  # label is a one-hot encoding vector of size 10
        assert type(data_set[0][1]) == np.ndarray
        assert data_set[0][1].shape == (10, 1)
        assert sum(data_set[0][1]) == 1  # only one bit in the label vector is 1, other 9 are 0
