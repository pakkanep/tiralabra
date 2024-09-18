# neural network object
import numpy as np


class NeuralNet():
    def __init__(self, layers:list):
        self.layers = len(layers)
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]



    def stochastic_gradient_descent(self):
        pass
        """
        call function minibatch 

        """

    def neuralnet_output(self, x):
        activations = x
        for bias, weight in zip(self.biases, self.weights):
            activations = sigmoid(np.dot(weight, activations) + bias)
        
        return activations


    def mini_batch(self, mini_batch, learning_rate):

        grad_biases = [np.zeros(b.shape) for b in self.biases] # sum of the cost func
        grad_weights = [np.zeros(w.shape) for w in self.weights] # same goes for weights
        

        for X, Y in mini_batch:
            nudges_b, nudges_w = self.backpropagation(X, Y)
            # call back propagation that calculates and returns the gradients
            # increase or decrease the weights and biases to minimize the cost func 
            grad_biases = [gb + nudge_b for gb, nudge_b in zip(grad_biases, nudges_b)]
            grad_weights = [gw + nudge_w for gw, nudge_w in zip(grad_biases, nudges_w)]

        
        m = len(mini_batch)
        
        self.weights = [w - (learning_rate / m) * grad_w for w, grad_w in zip(self.weights, grad_weights)]
        self.biases = [b - (learning_rate / m) * grad_b for b, grad_b in zip(self.biases, grad_biases)]



    def backpropagation(self, x, y):

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

