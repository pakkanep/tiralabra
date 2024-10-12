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
        """
        Passes input x trough the net and compares the output to y for all (x, y) pairs in data.

        Args:
            data (list) 
            training (bool)

        Returns:
            int: amount of correctly classified digits.
        """

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
        Searches the local minimum for the cost function

        Args:
            learning_rate (float) hyperparameter
            rounds (int)
            batch_size (int)
            training_data ()
            test_data ()
            validation_data ()
            show_learning_progress ()

        Returns:
            None
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
                    print("total cost validation data:",
                        self.total_cost(validation_data, validation=True)
                        )


    def neuralnet_output(self, x):
        """
        Calculates the output of the network with given input x
        
        Args:
            x (numpy.ndarray) vector): 28x28x pixel image reshaped to a (784, 1) vector.

        Returns:
            numpy.ndarray: (10, 0) vector.
        """
        activations = x
        for bias, weight in zip(self.biases, self.weights):
            activations = sigmoid(np.dot(weight, activations) + bias)
        return activations


    def cost_function(self, x, y):
        """
        Calculates the Mean squared error.

        Args:
            x (numpy.ndarray): (784, 1) vector.
            y (numpy.ndarray): (10, 1) vector.

        Returns:
            float: cost of one input-target pair (x,y).
        """

        y_hat = self.neuralnet_output(x)
        return 0.5 * np.sum((y_hat-y)**2)

    def total_cost(self, data, testing=False, validation=False):
        """
        Calculates total cost of all the input-target pairs (x,y) in the data set

        Args:
            data (list of (x, y)): x is the input and y is the correct output.
            testing or validation (bool): tells if y in data needs to be reshaped.

        Returns:
            float: total cost for the dataset.
        """

        cost = 0.0
        for x, y in data:
            if testing == True or validation == True:
                y = vectorize(y)
            cost += self.cost_function(x,y) / len(data)

        return cost


    def mini_batch(self, mini_batch, learning_rate):
        """
        Modifies the weights and biases in the network for one training set

        Args:
            mini_batch (list): list of input-target pairs (x,y).
            learning_rate (float): controls how much the models weights are adjusted with respect to the error during training.

        Returns:
            None.
        """
        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose() 
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()

        grad_biases, grad_weights = self.backpropagation(x, y)

        m = len(mini_batch)
        self.weights = [w - (learning_rate / m) * grad_w for w, grad_w in zip(self.weights, grad_weights)]
        self.biases = [b - (learning_rate / m) * grad_b for b, grad_b in zip(self.biases, grad_biases)]


    def feedforward(self, a):
        """
        Calculates layers activations and weighted input.

        Args:
            a (numpy.asarray): (784, 10) shape matrix

        Returns:
            list: all the activations for each layer.
            list: all the weighted inputs for all neurons 
        """
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
    
        Args:
            x (numpy.ndarray): (784, 10) shape matrix, that contains all inputs.
            y (numpy.ndarray): (10, 10) shape matrix, that contains all outputs.

        Returns:
            (tuple): gradient_biases, gradient_weights
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
    """
    Compute the sigmoid function for the input z.

    Args:
        z (numpy.ndarray): The input value or array for which to compute the sigmoid function.
    
    Returns:
        numpy.ndarray: The computed sigmoid value(s), ranging between 0 and 1.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Compute the derivative of the sigmoid function (also known as sigmoid prime).

    Args:
        z (numpy.ndarray): The input value or array for which to compute the derivative of the sigmoid function.

    Returns:
        numpy.ndarray: The computed derivative of the sigmoid function.
    """
    return sigmoid(z) * (1 - sigmoid(z))


def un_vectorize(y):
    """
    Convert a vector back to its original integer form.

    Args:
        y (numpy.ndarray): A (10, 1) shape vector, where one value is 1 and the rest are 0.

    Returns:
        int: The index (0-9) where the value is 1, representing the original class/label.
    """
    return np.where(y == 1)[0]


def vectorize(y):
    """
    Convert an integer label into a (10, 1) shape vector.

    Args:
        y (int): The index (0-9) that corresponds to the target class/label.

    Returns:
        numpy.ndarray: A (10, 1) one-hot encoded vector with 1 at the given index `y` and 0s elsewhere.
    """
    result = np.zeros((10, 1))
    result[y] = 1.0
    return result
