import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, roc_auc_score, roc_curve


def plot_precision_recall_curve(classification_result, **kwargs):
    print('precision_recall_fscore_support')
    print(precision_recall_fscore_support(
        classification_result.target, classification_result.predicted))

    precision, recall, thresholds = precision_recall_curve(
        classification_result.target, classification_result.predicted_proba)

    dots = np.arange(0, 1, 0.1)
    plt.scatter(dots, dots)

    plt.xlabel('precision', fontsize=23)
    plt.ylabel('recall', fontsize=23)
    plt.plot(precision, recall, **kwargs)

    return precision, recall, thresholds


def plot_roc_curve(classification_result):
    print('roc auc')
    print(roc_auc_score(classification_result.target, classification_result.predicted_proba))

    fpr, tpr, thresholds = roc_curve(classification_result.target, classification_result.predicted_proba)

    plt.xlabel('tpr', fontsize=23)
    plt.ylabel('fpr', fontsize=23)
    plt.plot(fpr, tpr)

    return tpr, fpr, thresholds


def plot_proba_distribution(classification_result):
    plt.title('predict_proba distribution', fontsize=23)
    plt.hist(classification_result.predicted_proba[classification_result.target],
             color='r', alpha=0.5, bins=100, density=True, label='positive')
    plt.hist(classification_result.predicted_proba[~classification_result.target],
             color='b', alpha=0.5, bins=100, density=True, label='negative')
    plt.legend(fontsize=23)


def show_classification_result(classification_result):
    for plotter in [plot_precision_recall_curve, plot_proba_distribution, plot_roc_curve]:
        plt.figure(figsize=(20, 10))

        plotter(classification_result)

        plt.show()
