#neural network object

class NeuralNet():
    def __init__(self, layers:list):
        pass


    def backpropagation(x, y):
        pass
        """
            Computes the gradient of the cost function for a single training example

            1. Input:
                jtn.

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

    def gradient_descent():
        pass
        """
            Compute the gradient for many training examples
            
            For each layer update the WEIGHTS with formula: w^l - n/m * sum(outputerror^x,l * (a^x,(l-1)^T)
            where:
                w^l = weights of the layer
                n = learning rate
                m = size of minibatch
                outputerror^x,l = output error of layer l in vector form
                (a^x,(l-1)^T = Transpose of activations of neurons of training example x on layer l-1 in matrix form

        """



