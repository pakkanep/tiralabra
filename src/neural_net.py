# neural network object
import random
import numpy as np

class NeuralNet():
    def __init__(self, layers:list):
        self.sizes = layers
        self.n_layers = len(layers)
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    def accuracy(self, data, training=False):
        if training == True:
            results = [(np.argmax(self.neuralnet_output(x)), un_vectorize(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.neuralnet_output(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy


    def stochastic_gradient_descent(
            self,
            learning_rate,
            rounds,
            batch_size,
            training_data,
            test_data=False,
            validation_data=False,
            show_learning_progress=True
        ):
        """
        Searches the local minimum for the cost func
        """
        
        n = len(training_data)
        for r in range(rounds):
            print("round: ",r, "/", rounds)
            random.shuffle(training_data)
            for idx in range(0, n, batch_size):
                minibatch = training_data[idx:idx+batch_size]
                self.mini_batch(minibatch, learning_rate)


            if show_learning_progress:    
                training_accuracy = self.accuracy(training_data, training=True)
                print("Training accuracy:", training_accuracy, "/", len(training_data))
                print("Total cost training data:", self.total_cost(training_data))

                if test_data:
                    test_accuracy = self.accuracy(test_data)
                    print("Test accuracy:", test_accuracy, "/", len(test_data))
                    print("Total cost test data:", self.total_cost(test_data, testing=True))

                if validation_data:
                    validation_accuracy = self.accuracy(validation_data)
                    print("Validation accuracy:", validation_accuracy)
                    print("total cost validation data:", self.total_cost(validation_data, validation=True))


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
        return 0.5 * np.sum((y_hat-y)**2)

    def total_cost(self, data, testing=False, validation=False):
        """
            Return the total cost for the data set
        """
        cost = 0.0
        for x, y in data:
            if testing == True or validation == True:
                y = vectorize(y)
            cost += self.cost_function(x,y) / len(data)

        return (cost)


    def mini_batch(self, mini_batch, learning_rate):
        """

        """
        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose() 
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()

        grad_biases, grad_weights = self.backpropagation(x, y)

        m = len(mini_batch)
        self.weights = [w - (learning_rate / m) * grad_w for w, grad_w in zip(self.weights, grad_weights)]
        self.biases = [b - (learning_rate / m) * grad_b for b, grad_b in zip(self.biases, grad_biases)]


    def feedforward(self, a):
        z_vectors = []
        activations_list = [a]
        activation = a

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_vectors.append(z)
            activation = sigmoid(z)
            activations_list.append(activation)

        return (z_vectors, activations_list)


    def backpropagation(self, x, y):
        """
        Calculates the gradient vectors
        """

        z_vectors, activations_list = self.feedforward(x)

        grad_biases = [0 for _ in self.biases]
        grad_weights = [0 for _ in self.weights]

        C_x = activations_list[-1] - y
        delta = C_x * sigmoid_prime(z_vectors[-1]) # (δ^L)

        grad_biases[-1] = delta.sum(1).reshape([len(delta), 1])
        grad_weights[-1] = np.dot(delta, activations_list[-2].transpose())


        # backpropagation loop

        for layer in range(2, self.n_layers):
            zl = z_vectors[-layer]
            derivative = sigmoid_prime(zl)

            delta = np.dot(self.weights[-layer+1].transpose(), delta) * derivative

            grad_biases[-layer] = delta.sum(1).reshape([len(delta), 1])
            grad_weights[-layer] = np.dot(delta, activations_list[-layer-1].transpose())

        return (grad_biases, grad_weights)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def un_vectorize(y):
    return np.where(y == 1)[0]

def vectorize(y):
    result = np.zeros((10, 1))
    result[y] = 1.0
    return result
