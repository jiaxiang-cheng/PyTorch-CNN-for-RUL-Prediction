import numpy as np
from target_rul import target_rul
import torch
from sklearn.model_selection import train_test_split


def train_val_prepare(max_cycle, idx, X_ss, calcul, nf, ns):

    train_img = []
    train_y = []
    id_engine = 1
    i = 0
    id_engine_end = max_cycle[id_engine - 1] - 1
    id_engine_start = 0

    while i <= ns - 15:

        img = X_ss[i:i + 15, ]
        img = img.astype('float32')
        train_img.append(img)

        train_y.append(target_rul(max_cycle[id_engine - 1], idx[i + 14, 1], calcul))

        i = i + 1
        if i + 14 <= ns - 1 and int(idx[i + 14, 0]) != id_engine:
            id_engine_start += max_cycle[id_engine - 1]
            i = i + 14
            id_engine += 1
            id_engine_end += max_cycle[id_engine - 1]

        # print("No.", id_engine, "No.", i, "th instance")

    # converting the list to numpy array
    train_x = np.array(train_img)
    # defining the target
    train_y = np.array(train_y)
    # train_x.shape
    # train_y.shape

    # create validation set
    # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)
    # (train_x.shape, train_y.shape), (val_x.shape, val_y.shape)

    # converting training images into torch format
    train_x = train_x.reshape(train_x.shape[0], 1, 15, nf)
    train_x = torch.from_numpy(train_x)

    # converting the target into torch format
    train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y)

    # shape of training data
    # train_x.shape, train_y.shape

    # # converting validation images into torch format
    # val_x = val_x.reshape(val_x.shape[0], 1, 15, nf)
    # val_x = torch.from_numpy(val_x)
    #
    # # converting the target into torch format
    # val_y = val_y.astype(int)
    # val_y = torch.from_numpy(val_y)

    # shape of validation data
    # val_x.shape, val_y.shape

    return train_x, train_y  # val_x, val_y


def test_prepare(Xt_ss, idx_t, nf, ns_t):
    test_img = []
    id_engine_t = 1
    i_t = 0
    while i_t <= ns_t - 15:
        img_t = Xt_ss[i_t:i_t + 15, ]
        img_t = img_t.astype('float32')
        test_img.append(img_t)
        i_t = i_t + 1
        if i_t + 14 <= ns_t - 1:
            if int(idx_t[i_t + 14, 0]) != id_engine_t:
                i_t = i_t + 14
                id_engine_t += 1
        # print("No.", id_engine_t, "No.", i_t, "th instance")

    # converting the list to numpy array
    test_x = np.array(test_img)
    # test_x.shape

    # converting training images into torch format
    test_x = test_x.reshape(test_x.shape[0], 1, 15, nf)
    test_x = torch.from_numpy(test_x)
    # test_x.shape
    return test_x
