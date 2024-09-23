from neural_net import NeuralNet
import load_data as ld
import numpy as np




if __name__ == "__main__":
    sizes = [784, 10, 10]
    neuroverkko = NeuralNet(sizes)
    training, validation, testing = ld.load_data()
    learning_rate = 3.0
    
    
    # neuroverkko.stochastic_gradient_descent(
    #     learning_rate,
    #     5,
    #     training,
    #     10,
    #     validation
    #     )
    print(10**-7)

    

    