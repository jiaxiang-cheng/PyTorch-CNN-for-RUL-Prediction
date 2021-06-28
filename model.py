from cnn_architecture import Net
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, MSELoss, Sequential, Conv2d, MaxPool2d, AvgPool2d, Module, \
    Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD


def model(nf):
    # defining the model
    model = Net(nf)
    # defining the optimizer
    # optimizer = SGD(model.parameters(), lr=0.007, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=0.01)
    # defining the loss function
    criterion = CrossEntropyLoss()
    # criterion = MSELoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    return model, optimizer, criterion
