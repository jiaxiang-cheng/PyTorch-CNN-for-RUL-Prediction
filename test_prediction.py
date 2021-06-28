import torch
import numpy as np


def test_prediction(model, test_x):
    # generating predictions for test set
    with torch.no_grad():
        output = model(test_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    return predictions
