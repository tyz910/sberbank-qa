""" Official evaluation script for the SDSJ dataset. """
from collections import Counter
import numpy as np
import re


def normalize_answer(text):
    """Lower text and remove punctuation and extra whitespace."""
    return ' '.join(re.findall(r"\w+", text)).lower()


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score_mean(predictions, ground_truths):
    return np.mean([f1_score(pred, gt) for pred, gt in zip(predictions, ground_truths)])
