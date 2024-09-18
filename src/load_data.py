import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open("src/data/mnist.pkl.gz", "rb")
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_data, val_data, test_data = u.load()
    f.close()
    
    training_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    training_results = [vectorized_result(y) for y in train_data[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    validation_data = list(zip(validation_inputs, val_data[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    testing_data = list(zip(test_inputs, test_data[1]))

    return (training_data, validation_data, testing_data)

def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e