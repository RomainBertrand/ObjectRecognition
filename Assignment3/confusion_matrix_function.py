'''
Original here: https://gist.githubusercontent.com/cbernecker/a7384c3394ea4e50ec68e7201a206741/raw/6338c24ce610dd45c54293e0f8a7f20878040dc3/confusion_matrix_function.py
'''

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

def createConfusionMatrix(loader, net, classes, device):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()
