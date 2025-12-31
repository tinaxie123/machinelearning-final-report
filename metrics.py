import numpy as np


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix

    Args:
        y_true: true labels, shape (n_samples,)
        y_pred: predicted labels, shape (n_samples,)

    Returns:
        cm: confusion matrix, shape (n_classes, n_classes)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    cm = np.zeros((n_classes, n_classes), dtype=int)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1

    return cm


def precision_recall_fscore(y_true, y_pred, average='macro', labels=None):
    """
    Compute precision, recall and F-score

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        y_true: true labels
        y_pred: predicted labels
        average: 'macro', 'weighted', 'micro', or None
        labels: list of labels to compute metrics for

    Returns:
        precision, recall, fscore: metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    n_classes = len(labels)

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    fscore = np.zeros(n_classes)
    support = np.zeros(n_classes)

    for i, label in enumerate(labels):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        support[i] = np.sum(y_true == label)

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision[i] + recall[i] > 0:
            fscore[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            fscore[i] = 0.0

    if average == 'macro':
        return np.mean(precision), np.mean(recall), np.mean(fscore)
    elif average == 'weighted':
        total = np.sum(support)
        weights = support / total if total > 0 else np.ones(n_classes) / n_classes
        return (np.sum(precision * weights),
                np.sum(recall * weights),
                np.sum(fscore * weights))
    elif average == 'micro':
        tp_total = np.sum([np.sum((y_true == label) & (y_pred == label)) for label in labels])
        fp_total = np.sum([np.sum((y_true != label) & (y_pred == label)) for label in labels])
        fn_total = np.sum([np.sum((y_true == label) & (y_pred != label)) for label in labels])

        prec_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        rec_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro) if (prec_micro + rec_micro) > 0 else 0.0

        return prec_micro, rec_micro, f1_micro
    else:
        return precision, recall, fscore


def f1_score(y_true, y_pred, average='macro', labels=None):
    """
    Compute F1 score

    Args:
        y_true: true labels
        y_pred: predicted labels
        average: 'macro', 'weighted', 'micro', or None
        labels: list of labels

    Returns:
        f1: F1 score(s)
    """
    _, _, f1 = precision_recall_fscore(y_true, y_pred, average=average, labels=labels)
    return f1


def classification_report(y_true, y_pred, digits=4, target_names=None):
    """
    Generate classification report

    Args:
        y_true: true labels
        y_pred: predicted labels
        digits: number of digits for formatting
        target_names: optional list of label names

    Returns:
        report: formatted string report
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))

    if target_names is None:
        target_names = [str(label) for label in labels]

    precision, recall, fscore = precision_recall_fscore(y_true, y_pred, average=None, labels=labels)
    support = np.array([np.sum(y_true == label) for label in labels])

    headers = ['precision', 'recall', 'f1-score', 'support']

    longest_label = max(len(name) for name in target_names)
    longest_label = max(longest_label, len('accuracy'))
    width = max(longest_label, len(headers[0]))

    header_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = header_fmt.format('', *headers, width=width)
    report += '\n\n'

    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'

    for i, label_name in enumerate(target_names):
        report += row_fmt.format(label_name,
                                 precision[i],
                                 recall[i],
                                 fscore[i],
                                 int(support[i]),
                                 width=width,
                                 digits=digits)

    report += '\n'

    accuracy = np.sum(y_true == y_pred) / len(y_true)
    report += row_fmt.format('accuracy',
                             accuracy,
                             accuracy,
                             accuracy,
                             int(np.sum(support)),
                             width=width,
                             digits=digits)

    macro_prec, macro_rec, macro_f1 = precision_recall_fscore(y_true, y_pred, average='macro')
    report += row_fmt.format('macro avg',
                             macro_prec,
                             macro_rec,
                             macro_f1,
                             int(np.sum(support)),
                             width=width,
                             digits=digits)

    weighted_prec, weighted_rec, weighted_f1 = precision_recall_fscore(y_true, y_pred, average='weighted')
    report += row_fmt.format('weighted avg',
                             weighted_prec,
                             weighted_rec,
                             weighted_f1,
                             int(np.sum(support)),
                             width=width,
                             digits=digits)

    return report
