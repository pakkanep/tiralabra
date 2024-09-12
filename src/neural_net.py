#neural network object
import numpy as np

class NeuralNet():
    def __init__(self, layers:list):
        self.weights = []
        self.biases = []


    def gradient_descent():
        pass
        """
        call function minibatch 

        """

    def mini_batch(mini_batch_size):
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
            #call back propagation that returns the gradient of the cost func
            pass

    def backpropagation(x, y):
        pass
        """
            Computes the gradient of the cost function for a single training example
            
            returns:

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
                compute the vector del^L = prime_sigmoid(z^L)

            4. Backpropagate the error:
                For each  l=L-1,L-2,...., 2

            5. Output:
                The gradient of the cost func is given by:
                nabla_C / nabla_weight between kth neuron from l-1 and jth neuron from l

                nabla_C / nabla_bias jth bias layer l
        
        """



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))



