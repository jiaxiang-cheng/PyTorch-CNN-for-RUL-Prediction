from train_function import train_function


def train(n_epochs, model, train_x, train_y, val_x, val_y, optimizer, criterion):
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    # training the model
    for epoch in range(n_epochs):
        train_function(epoch, model, train_x, train_y, val_x, val_y, optimizer, criterion, train_losses, val_losses)

    return train_losses, val_losses
