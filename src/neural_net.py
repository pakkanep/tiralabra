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

    def mini_batch(self, mini_batch_size):
        """
            Compute the gradient for many training examples
            
            For each layer update the WEIGHTS with formula: w^l - n/m * sum(outputerror^x,l * (a^x,(l-1)^T)
            where:
                w^l = weights of the layer
                n = learning rate
                m = size of minibatch
                outputerror^x,l = output error of layer l in vector form
                (a^x,(l-1)^T = Transpose of activations of neurons of training example x on layer l-1 in matrix form

            For each layer update the biases with formula: b^l - n/m * sum(δ^l)
            where:
                δ^l = error vectors

        """
        

        for do_something in range(mini_batch_size):
            # call back propagation that calculates and returns the gradients
            # update the weight and biases in the network
            pass



    def backpropagation(self, x, y):
        """
            Computes the gradient of the cost function for a single training example
            
            returns: gradients

            1. Input:
                x = 28x28 pixel image changed to a vector of shape (784, 1).
                the vector items are the first layers(input layers) neurons activation values between 0.0 - 1.0
                y = desired output which we then use to calculate the output error

            2. Feed forward:
                For each l = 2, 3,..., L compute
                z^l = w^l * a^l-1 + b^l 
                a^l = sigmoid(z^l) 

            Compute the error vectors δ^l backward, starting from the final layer.    
            
            3. Output error:
                compute the vector delta (δ^L) = prime_sigmoid(z^L)

            4. Backpropagate the error:
                For each  l=L-1,L-2,...., 2

            5. Output:
                nabla_biases: layer by layer list where each array represents the gradient of the cost function with respect to the biases for a layer.
                nabla_weights = layer by layer list where each array represents the gradient of the cost function with respect to the weights for a layer

        
        """
        # feed forward loop
        activations = [x] # x contains the activations from the input layer as vector shape of 784,
        weighted_inputs = [] # save weighted inputs z^l of all layers
        activation = activations[0] # sets the input layer activations to start counting the activations

        for w, b in zip(self.weights, self.biases): # not including the input layer
            weighted_input = np.dot(w, activation) + b # z^l = w^l * a^l-1 + b^l 
            weighted_inputs.append(weighted_input)
            activation = sigmoid(weighted_input) # a^l = sigmoid(z^l)
            activations.append(activation) 
            
        
        # calculate the error of last layer to be able to calc the error of previous layers
        grad_biases = [np.zeros(x.shape) for x in self.biases]
        grad_weights = [np.zeros(x.shape) for x in self.weights]

        delta = (activations[-1] - y) * sigmoid_prime(weighted_input[-1]) # (δ^L)
        grad_biases[-1] = delta
        grad_weights[-1] = np.dot(delta, activations[-2].transpose())


        # backpropagation loop
        
        for layer in range(2, self.layers): # from final layer to first layer but not input layer
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sigmoid_prime(weighted_inputs[-layer]) 
            grad_biases[-layer] = delta
            grad_weights[-layer] = np.dot(delta, activations[-layer-1].transpose())

        return (grad_biases, grad_weights)
    



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

