import unittest
from neural_net import NeuralNet

class TestNeuralNet(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
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

    #def test_order_of_samples_in_batch_doesnt_affect_output
    #def test_all_network_layers_change_after_each_optimizer_step
