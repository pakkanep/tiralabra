"""
neural_net.py

Stochastic gradient descent learning algorithm for a feedforward neural network. 
Gradients are calculated using backpropagation.
"""
import random
import numpy as np

class NeuralNet():
    """
        Attributes:
        methods:
    """
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

        if training is True:
            results = [(np.argmax(self.neuralnet_output(x)), un_vectorize(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.neuralnet_output(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x.item() == y.item()) for (x, y) in results)

        return result_accuracy

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def stochastic_gradient_descent(self, learning_rate, rounds,
            batch_size, data, show_learning_progress=True
        ):
        """
        Searches the local minimum for the cost function

        Args:
            learning_rate (float) hyperparameter
            rounds (int) n epochs of training
            batch_size (int) size of minibatch means n input-target pairs
            data (tuple) training_data, validation_data, test_data
            show_learning_progress (bool) True if learning progress is monitored

        Returns:
            None
        """
        training_data = data[0]
        n = len(training_data)
        for r in range(rounds):
            print("\nround: ",r, "/", rounds)
            random.shuffle(training_data)
            for idx in range(0, n, batch_size):
                self.mini_batch(training_data[idx:idx+batch_size], learning_rate)

            if show_learning_progress:
                self.print_progress(data)



    def neuralnet_output(self, x):
        """
        Calculates the output of the network with given input x
        
        Args:
            x (numpy.ndarray) vector): 28x28x pixel image reshaped to a (784, 1) vector.

        Returns:
            numpy.ndarray: (10, 1) vector.
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

    def total_cost(self, data, convert=False):
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
            if convert is True:
                y = vectorize(y)
            cost += self.cost_function(x,y) / len(data)

        return cost


    def mini_batch(self, mini_batch, learning_rate):
        """
        Modifies the weights and biases in the network for one training set

        Args:
            mini_batch (list): list of input-target pairs (x,y).
            learning_rate (float): controls how much the models weights
            are adjusted with respect to the error during training.

        Returns:
            None.
        """
        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose()
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()

        grad_biases, grad_weights = self.backpropagation(x, y)

        m = len(mini_batch)
        self.weights = [w - (learning_rate / m) * grad_w for
                        w, grad_w in zip(self.weights, grad_weights)]

        self.biases = [b - (learning_rate / m) * grad_b for
                       b, grad_b in zip(self.biases, grad_biases)]


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

        output_differences = activations_list[-1] - y
        delta = output_differences * sigmoid_prime(z_vectors[-1]) # (δ^L)

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

    def print_progress(self, data):
        """
        Prints the learning progress of the network

        Args:
            data (tuple): training_data, validation_data, test_data

        Returns:
            None
        """
        training_data, validation_data, test_data = data

        train_acc = self.accuracy(training_data[:10000], training=True)
        print(
            f"{'Training accuracy:':<20} {train_acc:>6d}/{len(training_data[:10000]):<6d}"\
            f"  Cost: {self.total_cost(training_data):>20.10f}"
        )

        if test_data:
            test_accuracy = self.accuracy(test_data)
            print(
                f"{'Test accuracy:':<20} {test_accuracy:>6d}/{len(test_data):<6d}"\
                f"  Cost: {self.total_cost(test_data, convert=True):>20.10f}"
            )

        if validation_data:
            valid_acc = self.accuracy(validation_data)
            print(
                f"{'Validation accuracy:':<20} {valid_acc:>6d}/{len(validation_data):<6d}"\
                f"  Cost: {self.total_cost(validation_data, convert=True):>20.10f}"
            )


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
        z (numpy.ndarray): The input value or array for which to compute 
        the derivative of the sigmoid function.

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
        numpy.ndarray: A (10, 1) one-hot encoded vector with 1 at the given
        index `y` and 0s elsewhere.
    """
    result = np.zeros((10, 1))
    result[y] = 1.0
    return result
