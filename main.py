"""RUL Prediction with CNN"""

from torch import nn, optim
from torch.autograd import Variable

import evaluation
from dataset_prepare import *
from load_data import *
from model import *
from test_prediction import *

N_EPOCH = 250


def train(n_epochs, model, train_x, train_y, test_x, max_cycle_t, y_test):
    rmse_history = []
    for epoch in range(1, n_epochs + 1):
        model.train()

        # getting the training set
        x_train, y_train = Variable(train_x), Variable(train_y)

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = model(x_train)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
        optimizer.step()

        if epoch % 1 == 0:
            # prediction on testing dataset
            pred = test_prediction(model, test_x)
            # evaluate the prediction accuracy
            _, rmse, score = evaluation.scoring(pred, max_cycle_t, y_test)
            print('Epoch :', epoch, '\t', 'loss :', round(loss_train.item(), 3), '\t',
                  "RMSE =", rmse, '\t', "Score =", score)
            rmse_history.append(rmse)

    return rmse_history


if __name__ == "__main__":
    # loading data
    train_raw, test_raw, max_cycle, max_cycle_t, y_test = load_data_FD001()
    X_ss, idx, Xt_ss, idx_t, nf, ns, ns_t = get_info(train_raw, test_raw)

    # prepare training and validation dataset
    train_x, train_y = train_val_prepare(max_cycle, idx, X_ss, "linear", nf, ns)
    # prepare testing dataset
    test_x = test_prepare(Xt_ss, idx_t, nf, ns_t)

    # initialize the cnn model
    model = CNN1(nf)
    # defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # defining the loss function
    criterion = nn.CrossEntropyLoss()

    # train the model
    rmse_history = train(N_EPOCH, model, train_x, train_y, test_x, max_cycle_t, y_test)

    # prediction on testing dataset
    predictions = test_prediction(model, test_x)
    # evaluate the prediction accuracy
    result, rmse, score = evaluation.scoring(predictions, max_cycle_t, y_test)
    evaluation.visualization(y_test, result, rmse)
    print(min(rmse_history))
