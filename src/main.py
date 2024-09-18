from neural_net import NeuralNet
import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import random

def load_test_data():
    data = np.load("src/data/100_test_data.npz")
    test_inputs = data["inputs"]
    test_labels = data["labels"]
    test_inputs = [x.reshape((784, 1)) for x in test_inputs]
    test_data = list(zip(test_inputs, test_labels))

    return test_data

# def load_train_data():
#     training, validation, testing = ld.load_data()

#     return (training, validation, testing)


if __name__ == "__main__":
    sizes = [784, 50, 10]
    neuroverkko = NeuralNet(sizes)
    training, validation, testing = ld.load_data()
    test_data = load_test_data()
    learning_rate = 3.0
    batch_size = 10
    n = len(training)

    rundit =0
    for round in range(10):
        random.shuffle(training)
        for idx in range(0, n, batch_size):
            rundit+=1
            mini_batch = training[idx:idx+batch_size]
            neuroverkko.mini_batch(mini_batch, learning_rate)
        
        print(rundit)
        summa = 0
        for testi in testing:
            if np.argmax(neuroverkko.neuralnet_output(testi[0])) == testi[1]:
                summa += 1
        print(summa)
        print()

    

    