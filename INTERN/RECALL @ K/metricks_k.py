from typing import List
import numpy as np


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    # Сортируем картинки по предсказанному скору в порядке убывания
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Вычисляем Recall @ K
    true_positives = 0
    total_positives = sum(labels)

    for i in range(k):
        if labels[sorted_indices[i]] == 1:
            true_positives += 1

    recall = true_positives / total_positives if total_positives > 0 else 0.0

    return recall


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    # Сортируем картинки по предсказанному скору в порядке убывания
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Вычисляем Precision @ K
    true_positives = 0

    for i in range(k):
        if labels[sorted_indices[i]] == 1:
            true_positives += 1

    precision = true_positives / k if k > 0 else 0.0

    return precision


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """Calculate the specificity@k metric."""

    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    _, labels = zip(*pairs)
    tn = sum(1 - np.array(labels[k:]))
    fp = sum(1 - np.array(labels[:k]))
    if all([tn == 0, fp == 0]):
        metric = 0
        return metric
    metric = tn / (tn + fp)
    return metric

def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    # Получаем Recall и Precision для K
    recall = recall_at_k(labels, scores, k)
    precision = precision_at_k(labels, scores, k)

    # Вычисляем F1-Score @ K
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1
