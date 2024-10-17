import load_data as ld
from neural_net import NeuralNet


def ui():
    print("Note that the ui does not do any error checking. Implemented for testing the network easily")

    while True:
        print("\nNeural Network Menu")
        print("1. Train network")
        print("2. Test network")
        print("3. Load existing network (added later)")
        print("4. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            train_network()
        elif choice == "2":
            test_network()
        elif choice == "3":
            print("Load existing network functionality will be added later.")
        elif choice == "4":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please select again.")

def train_network():
    layers = input("Enter network layers as comma-separated values (default: 784,100,10): ") or "784,100,10"
    epochs = input("Enter number of epochs (default: 10): ") or "10"
    batch_size = input("Enter mini-batch size (default: 10): ") or "10"
    learning_rate = input("Enter learning rate (default: 3.0): ") or "3.0"
    display_progress = input("Display progress during training? (yes/no, default: no): ") or "no"

    layer_sizes = list(map(int, layers.split(',')))
    epochs = int(epochs)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)

    print(f"\nTraining network with layers {layer_sizes},\
        {epochs} epochs, batch size {batch_size},\
        learning rate {learning_rate}.")

    training_data, validation_data, test_data = ld.load_data()
    net = NeuralNet(layer_sizes)
    if display_progress == "no":

        net.stochastic_gradient_descent(
                learning_rate=learning_rate,
                rounds=epochs,
                batch_size=batch_size,
                training_data=training_data,
                test_data=test_data,
                validation_data=validation_data,
                show_learning_progress=False
        )
    else:
        net.stochastic_gradient_descent(
                learning_rate=learning_rate,
                rounds=epochs,
                batch_size=batch_size,
                training_data=training_data,
                test_data=test_data,
                validation_data=validation_data,
                show_learning_progress=True
        )

def test_network():
    print("Testing network (functionality to be added).")

if __name__ == "__main__":
    ui()
