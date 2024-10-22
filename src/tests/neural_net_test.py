"""
Test module for neural network
"""
import random
import unittest
import numpy as np
from neural_net import NeuralNet
import load_data as ld


def load_data():
    """
    Loads the data for tests
    """
    training, validation, testing = ld.load_data()
    return training, validation, testing


class TestNeuralNet(unittest.TestCase):
    """
    Test class for testing neural network methods
    """
    def setUp(self):
        #np.random.seed(90)
        self.neural_net = NeuralNet([784, 20, 10])
        self.data = load_data()

    def test_constructor(self):
        """
        checks that the network gets constructed correctly
        """

    def test_accuarcy_with_convert(self):
        """
        Tests that Accuracy works with training data which needs converting target value y
        """
        training_data = self.data[0][:1000]
        output = self.neural_net.accuracy(training_data, training=True)
        self.assertEqual(type(output), int)

    def test_accuracy_without_convert(self):
        """
        Tests that accuracy works without the need of converting target value y
        """
        test_data = self.data[2][:1000]
        output = self.neural_net.accuracy(test_data)
        self.assertEqual(type(output), int)


    def test_neuralnet_output(self):
        """
        tests that the functions input/output is right shape and type
        """
        x = self.data[0][0][0] # <class 'numpy.ndarray'>(784, 1)
        output = self.neural_net.neuralnet_output(x)
        expected_out = np.empty((10, 1))
        self.assertEqual(type(output), type(expected_out))
        self.assertEqual(output.shape, expected_out.shape)


    def test_backpropagation(self):
        """
        Calculate numerical gradient and compare it to the analytical gradient
        that backprop calculates to ensure that it calculates it correctly.
        """
        x, y = self.data[0][0][0], self.data[0][0][1]
        epsilon = 1e-5
        nabla_w = self.neural_net.backpropagation(x, y)[1]

        num_nabla_w = [np.zeros(w.shape) for w in self.neural_net.weights]


        for l in range(len(self.neural_net.weights)):
            for i in range(self.neural_net.weights[l].shape[0]):
                for j in range(self.neural_net.weights[l].shape[1]):
                    org_w = self.neural_net.weights[l][i, j]

                    self.neural_net.weights[l][i, j] = org_w + epsilon
                    cost_plus = self.neural_net.cost_function(x, y)

                    self.neural_net.weights[l][i, j] = org_w - epsilon
                    cost_minus = self.neural_net.cost_function(x, y)

                    self.neural_net.weights[l][i, j] = org_w
                    num_nabla_w[l][i, j] = (cost_plus - cost_minus) / (2 * epsilon)

        rel_diffs = []

        for l in range(len(nabla_w)):
            difference = np.linalg.norm(
                num_nabla_w[l] - nabla_w[l]) / (np.linalg.norm(num_nabla_w[l]) +
                                                np.linalg.norm(nabla_w[l])
                )
            rel_diffs.append(difference)

        return self.assertGreater(10**-7, max(rel_diffs))


    def test_classification_accuracy_increases(self):
        """
        Tests that the classification increases after an epoch of minibatch
        updating weights and biases. Does two epochs so the comparison does
        not happen based on random data
        """
        training_data = self.data[0][:20000]
        validation_data = self.data[1]
        n = len(training_data)
        accuracy1, accuracy2 = 0.0, 0.0
        batch_size = 10
        learning_rate = 3.0
        for epoch in range(3):
            random.shuffle(training_data)
            for idx in range(0, n, batch_size):
                minibatch = training_data[idx:idx+batch_size]
                self.neural_net.mini_batch(minibatch, learning_rate)

            if epoch == 0:
                accuracy1 = self.neural_net.accuracy(validation_data)
            else:
                accuracy2 = self.neural_net.accuracy(validation_data)

        return self.assertGreater(accuracy2, accuracy1)



    def test_gradients_are_non_zero(self):
        """
        Tests that backpropagation does return nonzero gradients to ensure that
        data passes through the forward pass and at least some loss signal propagates
        backwards
        """
        training_data = self.data[0][:1]
        grad_b, grad_w = self.neural_net.backpropagation(training_data[0][0], training_data[0][1])

        grad_b_non_zero, grad_w_non_zero = True, True

        for grad in grad_w[1:]:
            if np.any(grad == 0) is True:
                grad_w_non_zero = False
                break

        for grad in grad_b:
            if np.any(grad == 0) is True:
                grad_b_non_zero = False
                break

        return self.assertTrue(grad_b_non_zero, grad_w_non_zero)

    def test_training_cost_decreases(self):
        """
        Checks that the cost decreases after couple of batches
        that ensures that the network works correctly overall
        """
        training_data = self.data[0][:20000]
        n = len(training_data)
        cost_epoch_1, cost_epoch_2 = 0.0, 0.0
        batch_size = 10
        learning_rate = 3.0
        for epoch in range(2):
            random.shuffle(training_data)
            for idx in range(0, n, batch_size):
                minibatch = training_data[idx:idx+batch_size]
                self.neural_net.mini_batch(minibatch, learning_rate)

            if epoch == 0:
                cost_epoch_1 = self.neural_net.total_cost(training_data[:5000])
            else:
                cost_epoch_2 = self.neural_net.total_cost(training_data[:5000])

        return self.assertGreater(cost_epoch_1, cost_epoch_2)


    def test_all_network_layers_change_after_each_optimizer_step(self):
        """
        Tests that all the layers changes meaning that weights
        and biases in each layer gets updated after a few batches
        to ensure all layers are used.
        This test uses more layers than others
        """
        self.multilayer = NeuralNet([784, 20, 20, 20, 10])
        ogw = self.multilayer.weights
        ogb = self.multilayer.biases
        training_data = self.data[0][:5000]
        checker = True
        n = len(training_data)
        batch_size = 10
        learning_rate = 2.0
        for epoch in range(3):
            random.shuffle(training_data)
            for idx in range(0, n, batch_size):
                minibatch = training_data[idx:idx+batch_size]
                self.multilayer.mini_batch(minibatch, learning_rate)

            for og, new in zip(ogb, self.multilayer.biases):
                val = np.all(np.equal(og, new))
                if val is True:
                    checker = False
                    break


            for og, new in zip(ogw, self.multilayer.weights):
                val = np.all(np.equal(og, new))
                if val is True:
                    checker = False
                    break

        return self.assertTrue(checker)


    def test_order_of_samples_in_batch_doesnot_affect_output(self):
        """
        Passes the same batch of data in different order two times to catch
        inconsistencies in how the model forward works with batching vs. individual samples.
        """
        result = True
        test_data = self.data[2]
        training_data = self.data[0][:5000]
        self.neural_net.mini_batch(training_data[:1000], 0.5)
        total_cost1 = round(self.neural_net.total_cost(test_data, convert=True))
        output1 = self.neural_net.accuracy(test_data)
        random.shuffle(test_data)
        total_cost2 = round(self.neural_net.total_cost(test_data, convert=True))
        output2 = self.neural_net.accuracy(test_data)
        if total_cost1 != total_cost2:
            result = False
        if output1 != output2:
            result = False

        return self.assertEqual(True, result)

    def test_network_overfits_to_a_small_dataset(self):
        pass
