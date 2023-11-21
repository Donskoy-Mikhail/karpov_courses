import numpy as np
from typing import List

def specificity_at_k(y_true, y_pred_prob, k):
    sorted_indices = np.argsort(y_pred_prob)[::-1]
    top_k_indices = sorted_indices[:k]
    top_k_predictions = y_true[top_k_indices]

    true_positives = np.sum(top_k_predictions == 1)
    num_positives = np.sum(y_true == 1)
    specificity = true_positives / num_positives

    return specificity

# Пример использования
y_true = np.array([1, 0, 0, 1, 1, 0])
y_pred_prob = np.array([0.9, 0.7, 0.2, 0.6, 0.8, 0.3])

k = 3
specificity_at_k_value = specificity_at_k(y_true, y_pred_prob, k)
print(f'Specificity @ {k} = {specificity_at_k_value}')

def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    # Сортируем картинки по предсказанному скору в порядке убывания
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[::-1]

    # Вычисляем Specificity @ K
    true_negatives = 0
    total_negatives = len(labels) - sum(labels)

    for i in range(k):
        if labels[sorted_indices[-(i+1)]] == 0:
            true_negatives += 1

    specificity = true_negatives / total_negatives

    return specificity

# Пример использования
y_true = np.array([1, 0, 0, 1, 1, 0])
y_pred_prob = np.array([0.9, 0.7, 0.2, 0.6, 0.8, 0.3])

k = 3
specificity_at_k_value = specificity_at_k(y_true, y_pred_prob, k)
print(f'Specificity @ {k} = {specificity_at_k_value}')