import numpy as np

from phishdet.thresholding import find_threshold_max_recall_under_fpr


def test_threshold_respects_fpr_and_maximizes_recall():
    # Construct probabilities where higher values correspond to positives
    y_true = np.array([0] * 95 + [1] * 5)
    # assign low probs to negatives, higher to positives but include overlap
    y_proba = np.concatenate([np.linspace(0.0, 0.2, 95), np.linspace(0.1, 0.9, 5)])
    threshold, metrics = find_threshold_max_recall_under_fpr(y_true, y_proba, max_fpr=0.01)
    preds = (y_proba >= threshold).astype(int)
    fp = int(((y_true == 0) & (preds == 1)).sum())
    tn = int(((y_true == 0) & (preds == 0)).sum())
    fpr = fp / (fp + tn)
    assert fpr <= 0.01 + 1e-6
    # At most one positive misclassified; recall should be maximized given constraint
    assert metrics["recall"] >= 0.6
