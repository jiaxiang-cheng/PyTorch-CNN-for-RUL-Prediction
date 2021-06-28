from load_data import *
from dataset_prepare import train_val_prepare, test_prepare
from model import model
from train import train
from test_prediction import test_prediction
from evaluation import evaluation

if __name__ == "__main__":
    # loading data
    train_raw, test_raw, max_cycle, max_cycle_t, y_test = load_data_FD001()
    X_ss, idx, Xt_ss, idx_t, nf, ns, ns_t = get_info(train_raw, test_raw)

    # prepare training and validation dataset
    train_x, train_y, val_x, val_y = train_val_prepare(max_cycle, idx, X_ss, "linear", nf, ns)
    # prepare testing dataset
    test_x = test_prepare(Xt_ss, idx_t, nf, ns_t)

    # initialize the cnn model
    model, optimizer, criterion = model(nf)

    # train the model
    train_losses, val_losses = train(150, model, train_x, train_y, val_x, val_y, optimizer, criterion)

    # visualize the training and validation loss
    # loss_visualization(train_losses, val_losses)

    # prediction on testing dataset
    predictions = test_prediction(model, test_x)
    # evaluate the prediction accuracy
    evaluation(predictions, max_cycle_t, y_test)
