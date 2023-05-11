import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
from io import StringIO
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    matplotlib.use('Agg')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    figg = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    imgdata = StringIO()
    figg.savefig(imgdata, format='svg')
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data


def model1():
    df = pd.read_csv("/Users/stepankozevnikov/PycharmProjects/Kursovaya/main/stds.csv")
    df = df.drop_duplicates().dropna()
    df.columns = map(lambda x: x.strip(), df.columns)
    df = df[df['Flow Bytes/s'] < 2 ** 31]
    df = df[df['Flow Packets/s'] < 2 ** 31]
    df['Label'] = df['Label'].replace({'BENIGN': 0, 'DDoS': 1})
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train), model.score(X_test, y_test))
    train_rf_predictions = model.predict(X_train)
    train_rf_probs = model.predict_proba(X_train)[:, 1]
    rf_predictions = model.predict(X_test)
    rf_probs = model.predict_proba(X_test)[:, 1]
    y_true = y_test
    y_pred = rf_predictions
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    data = plot_confusion_matrix(cm, classes=['BENIGN', 'DDoS'],
                          title='DDoS Confusion Matrix')
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)


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


def model2():
    df = pd.read_csv('stds.csv')
    df = df.drop_duplicates().dropna()
    df.columns = map(lambda x: x.strip(), df.columns)
    df = df[df['Flow Bytes/s'] < 2 ** 31]
    df = df[df['Flow Packets/s'] < 2 ** 31]
    df['Label'] = df['Label'].replace({'BENIGN': 0, 'DDoS': 1})
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train), model.score(X_test, y_test))
    arr = cross_val_score(model, X, y, cv=5)
    train_rf_predictions = model.predict(X_train)
    train_rf_probs = model.predict_proba(X_train)[:, 1]
    rf_predictions = model.predict(X_test)
    rf_probs = model.predict_proba(X_test)[:, 1]
    y_true = y_test
    y_pred = rf_predictions
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
