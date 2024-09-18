import unittest
import numpy as np
from neural_net import NeuralNet

def load_test_data():
    data = np.load("src/data/unit_tests_data.npz")
    test_inputs = data["inputs"]
    test_labels = data["labels"]
    test_inputs = [x.reshape((784, 1)) for x in test_inputs]
    test_data = list(zip(test_inputs, test_labels))

    return test_data


class TestNeuralNet(unittest.TestCase):
    def setUp(self):
        np.random.seed(90)
        self.neural_net = NeuralNet([784, 10, 10])

    def test_constructor(self):
        pass
        
    
    def test_backpropagation(self):
        pass

    def test_mini_batch(self):
        pass
        
    def test_gradients_are_non_zero_and_loss_decreases(self):
        pass

    def test_network_overfits_to_a_small_dataset(self):
        pass
    

    def test_hello_world(self):
        self.assertEqual("Hello world", "Hello world")

    #def test_order_of_samples_in_batch_doesnot_affect_output
    #def test_all_network_layers_change_after_each_optimizer_step
