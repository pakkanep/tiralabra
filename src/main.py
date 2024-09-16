from neural_net import NeuralNet
import numpy as np


if __name__ == "__main__":
    neuroverkko = NeuralNet([2, 3, 1])
    print(neuroverkko.weights)
    print()
    print(neuroverkko.biases)

    x = np.array([[0], [0]])
    y = np.array([[1]])
    w, b = neuroverkko.backpropagation(x, y)
    print("TULOS")
    print(w)
    print()
    print(b)