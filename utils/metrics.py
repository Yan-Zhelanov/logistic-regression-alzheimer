import numpy as np


def get_accuracy_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Get accuracy score.

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

    Returns:
        float: Accuracy score.
    """
    return np.mean(targets == predictions)


def get_precision_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Get precision score.

    The formula is as follows:
        precision = TP / (TP + FP),

        where:
            - TP is the number of true positives,
            - FP is the number of false positives.

    Args:
        targets: true labels
        predictions: predicted class

    Returns:
        float: Precision score.
    """
    true_positive = np.sum((targets == predictions)[targets == 1])
    false_positive = np.sum((targets != predictions)[targets == 0])
    return true_positive / (true_positive + false_positive)


def get_recall_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Get recall score.

    The formula is as follows:
        recall = TP / (TP + FN),

        where:
            - TP is the number of true positives,
            - FN is the number of false negatives.

    Args:
        targets: true labels
        predictions: predicted class

    Returns:
        float: Recall score.
    """
    true_positive = np.sum((targets == predictions)[targets == 1])
    false_negative = np.sum((targets != predictions)[targets == 1])
    return true_positive / (true_positive + false_negative)


def get_confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray,
) -> np.ndarray:
    """Get confusion matrix.

    Confusion matrix C with shape KxK:
        c[i, j] - number of observations known to be in class i and predicted
        to be in class j,
        where:
            - K is the number of classes.

    Args:
        targets: Labels.
        predictions: Predicted classes.

    Returns:
        np.ndarray: Confusion matrix.
    """
    count_classes = len(np.unique(targets))
    confusion_matrix = np.zeros((count_classes, count_classes))
    np.add.at(confusion_matrix, (targets, predictions), 1)
    return confusion_matrix


def get_precision_recall_curve(
    targets: np.ndarray, scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get precision-recall curve computation for different threshold.

    Args:
        targets: true labels
        scores: target scores

    Returns:
        precision (np.ndarray): precision values
        recall (np.ndarray): recall values
        thresholds (np.ndarray): thresholds
    """
    thresholds = np.unique(scores)[::-1]
    precisions = np.array([0])
    recalls = np.array([0])
    for threshold in thresholds:
        predictions = scores >= threshold
        count_predicted_positives = np.sum(predictions)
        if count_predicted_positives:
            precisions = np.append(
                precisions, get_precision_score(targets, predictions),
            )
        else:
            precisions = np.append(precisions, 0)
        recalls = np.append(recalls, get_recall_score(targets, predictions))
    np.append(precisions, 0)
    np.append(recalls, 1)
    precisions = np.flip(np.maximum.accumulate(np.flip(precisions)))
    return precisions, recalls, thresholds


def get_average_precision_score(
    targets: np.ndarray, scores: np.ndarray,
) -> float:
    """Compute Average Precision metric.

    Average precision is area under precision-recall curve:
        AP = Σ (R_n - R_{n-1}) * P_n,

        where:
            - P_n and R_n are the precision and recall at the n-th threshold

    Args:
        targets: True labels.
        scores: Target scores.

    Returns:
        float: Average precision score.
    """
    precisions, recalls, _ = get_precision_recall_curve(targets, scores)
    recall_deltas = np.diff(recalls)
    return np.sum(recall_deltas * precisions[1:])
