# neural network object
import random
import numpy as np


class NeuralNet():
    def __init__(self, layers:list):
        self.layers = len(layers)
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]



    def stochastic_gradient_descent(self, learning_rate, rounds,
                                    training_data, batch_size, validation_data):
        """
        Searches the local minimum for the cost func
        """
        n = len(training_data)
        for r in range(rounds):
            random.shuffle(training_data)
            for idx in range(0, n, batch_size):
                minibatch = training_data[idx:idx+batch_size]
                self.mini_batch(minibatch, learning_rate)
            valid_sum = 0
            for validi in validation_data:
                if np.argmax(self.neuralnet_output(validi[0])) == validi[1]:
                    valid_sum += 1
            print(valid_sum, "/", 10000)


    def neuralnet_output(self, x):
        """
        Calculates the output of the network with given input x
        """
        activations = x
        for bias, weight in zip(self.biases, self.weights):
            activations = sigmoid(np.dot(weight, activations) + bias)
        return activations


    def cost_function(self, x, y):
        """
        Calculates the MSE 
        """
        y_hat = self.neuralnet_output(x)
        return 0.5 * (np.sum(y_hat-y)**2)


    def mini_batch(self, mini_batch, learning_rate):
        """

        """

        grad_biases = [np.zeros(b.shape) for b in self.biases] # sum of the cost func
        grad_weights = [np.zeros(w.shape) for w in self.weights] # same goes for weights


        for x, y in mini_batch:
            nudges_b, nudges_w = self.backpropagation(x, y)
            #print(nudges_b[0].shape, nudges_b[1].shape)

            grad_biases = [gb + nudge_b for gb, nudge_b in zip(grad_biases, nudges_b)]
            grad_weights = [gw + nudge_w for gw, nudge_w in zip(grad_weights, nudges_w)]


        m = len(mini_batch)
        self.weights = [w - (learning_rate / m) * grad_w for w, grad_w in zip(self.weights, grad_weights)]
        self.biases = [b - (learning_rate / m) * grad_b for b, grad_b in zip(self.biases, grad_biases)]


    def backpropagation(self, x, y):
        """
        Calculates the gradient vectors
        """

        activations = [x] 
        z_vectors = []
        activation = x 

        for b, w in zip(self.biases, self.weights): # not including the input layer
            weighted_input = np.dot(w, activation) + b # z^l = w^l * a^l-1 + b^l 
            z_vectors.append(weighted_input)

            activation = sigmoid(weighted_input) # a^l = sigmoid(z^l)
            activations.append(activation) 


        grad_biases = [np.zeros(b.shape) for b in self.biases]
        grad_weights = [np.zeros(w.shape) for w in self.weights]

        C_x = activations[-1] - y
        delta = C_x * sigmoid_prime(z_vectors[-1]) # (δ^L)


        grad_biases[-1] = delta
        grad_weights[-1] = np.dot(delta, activations[-2].transpose())


        # backpropagation loop

        for layer in range(2, self.layers): # from final layer to first layer but not input layer
            zl = z_vectors[-layer]
            derivative = sigmoid_prime(zl)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * derivative

            grad_biases[-layer] = delta
            grad_weights[-layer] = np.dot(delta, activations[-layer-1].transpose())

        return (grad_biases, grad_weights)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
