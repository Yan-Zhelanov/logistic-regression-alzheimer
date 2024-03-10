import numpy as np


def accuracy_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Accuracy score.

    The formula is as follows:
        accuracy = (1 / N) Σ(i=0 to N-1) I(y_i == t_i),

        where:
            - N - number of samples,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i) - indicator function.
    Args:
        targets: true labels
        predictions: predicted class
    """
    # TODO: Implement computation of accuracy
    pass


def precision_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Precision score.

    The formula is as follows:
        precision = TP / (TP + FP),

        where:
            - TP is the number of true positives,
            - FP is the number of false positives.
    Args:
        targets: true labels
        predictions: predicted class
    """
    # TODO: Implement computation of precision
    pass


def recall_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Recall score.

    The formula is as follows:
        recall = TP / (TP + FN),

        where:
            - TP is the number of true positives,
            - FN is the number of false negatives.
    Args:
        targets: true labels
        predictions: predicted class
    """
    # TODO: Implement computation of recall
    pass


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray):
    """Confusion matrix.

    Confusion matrix C with shape KxK:
        c[i, j] - number of observations known to be in class i and predicted to be in class j,

        where:
            - K is the number of classes.

    Args:
        targets: labels
        predictions: predicted class
    """
    # TODO: Implement computation of confusion matrix
    pass


def precision_recall_curve(targets: np.ndarray, scores: np.ndarray):
    """Precision-Recall curve computation for different threshold.

    Args:
        targets: true labels
        scores: target scores

    Returns:
        precision (np.ndarray): precision values
        recall (np.ndarray): recall values
        thresholds (np.ndarray): thresholds
    """
    # TODO:
    #  1) Get thresholds - unique values of scores by descending order
    #  2) Add first precision and recall values as 0 and 0
    #  3) For every th_n compute precision and recall:
    #       - P_n = (Σ_i I(t_i == 1) * I(score_i >= th_n)) / (Σ_i I(score_i >= th_n))
    #       - R_n = (Σ_i I(t_i == 1) * I(score_i >= th_n)) / (Σ_i I(t_i == 1))
    #       - save P_n, R_n and th_n for every n
    #  4) Add last precision and recall values as 0 and 1, respectively
    #  5) For n from len(thresholds)-1 to 0:
    #       - P_{n-1} = max(P_n, P_{n-1})
    pass


def average_precision_score(targets: np.ndarray, scores: np.ndarray) -> float:
    """Computes Average Precision metric.

    Average precision is area under precision-recall curve:
        AP = Σ (R_n - R_{n-1}) * P_n,

        where:
            - P_n and R_n are the precision and recall at the n-th threshold
    Args:
        targets: true labels
        scores: target scores
    """
    # TODO: Implement computation of average precision
    pass
