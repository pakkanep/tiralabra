from neural_net import NeuralNet
import numpy as np
import matplotlib.pyplot as plt

def load_test_data():
    data = np.load("src/data/unit_tests_data.npz")
    test_inputs = data["inputs"]
    test_labels = data["labels"]
    test_inputs = [x.reshape((784, 1)) for x in test_inputs]
    test_data = list(zip(test_inputs, test_labels))

    return test_data

if __name__ == "__main__":
    sizes = [784, 30, 10]
    neuroverkko = NeuralNet(sizes)
    mini_batch = load_test_data()


    for rounds in range(3):

        grad_biases = [np.zeros(b.shape) for b in neuroverkko.biases]
        grad_weights = [np.zeros(w.shape) for w in neuroverkko.weights]
        
        for x, y in mini_batch:
            nudges_b, nudges_w = neuroverkko.backpropagation(x, y)
            grad_biases = [b + nudge_b for b, nudge_b in zip(grad_biases, nudges_b)]
            grad_weights = [w + nudge_w for w, nudge_w in zip(grad_biases, nudges_w)]

        neuroverkko.biases = [b - (0.01 / 3.0) * gb for b, gb in zip(neuroverkko.biases, grad_biases)]
        neuroverkko.weights = [w - (0.01 / 3.0) * gw for w, gw in zip(neuroverkko.weights, grad_weights)]
    

    