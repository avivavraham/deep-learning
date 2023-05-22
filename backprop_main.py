import backprop_data
import backprop_network
import matplotlib.pyplot as plt

LEARNING_RATE_MEAN = 0.1237
NUMBER_OF_LEARNING_RATES = 9
EPOCHS = 30
training_data, test_data = backprop_data.load(train_size=50000, test_size=1000)

net = backprop_network.Network([784, 40, 10])

net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)


def q1b():
    """
    this function is made for plotting test accuracy, train accuracy and train loss.
    you can do so by uncomment the relevant part of code for your objective.
    """
    test_accuracies = []
    training_losses = []
    train_accuracies = []
    for i in range(NUMBER_OF_LEARNING_RATES):
        network = backprop_network.Network([784, 40, 10])
        learning_rate = LEARNING_RATE_MEAN + 0.0001 * i
        test_accuracy, train_accuracy, training_loss = network.SGD(training_data, epochs=EPOCHS, mini_batch_size=10,
                                                                   learning_rate=learning_rate, test_data=test_data)
        test_accuracies.append(test_accuracy)
        train_accuracies.append(train_accuracy)
        training_losses.append(training_loss)
        plt.plot([j+1 for j in range(EPOCHS)], test_accuracies[i], label=f"learning rate of {learning_rate}")
        # plt.plot([j+1 for j in range(EPOCHS)], train_accuracies[i], label=f"learning rate of {learning_rate}")
        # plt.plot([j + 1 for j in range(EPOCHS)], training_losses[i], label=f"learning rate of {learning_rate}")
    plt.xlabel('epochs')
    # naming the y axis,
    plt.ylabel('loss')
    # plt.ylabel('accuracy')
    # plt.ylim((0, 1))
    plt.title('training loss across epochs')
    # plt.title('testing accuracy across epochs')
    # plt.title('training accuracy across epochs')
    plt.legend()
    plt.show()


# q1b()
