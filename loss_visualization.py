import matplotlib.pyplot as plt


def loss_visualization(train_losses, val_losses):
    # plotting the training and validation loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()
