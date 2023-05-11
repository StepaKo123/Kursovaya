import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
from io import StringIO


def rezlts():
    with open('/Users/stepankozevnikov/PycharmProjects/Kursovaya/main/modelfridman/data/result.txt', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    arr1 = np.array(lines)

    with open('/Users/stepankozevnikov/PycharmProjects/Kursovaya/main/modelfridman/data/test_data.csv', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    last_column = [line.split(',')[-1] for line in lines]
    arr2 = np.array(last_column)
    y_true = arr2
    y_pred = arr1
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_roc = auc(fpr, tpr)
    matplotlib.use('Agg')
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    data = imgdata.getvalue()

    context = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc_roc': data
    }
    return context
