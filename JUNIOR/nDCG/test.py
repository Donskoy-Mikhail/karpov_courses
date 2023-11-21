import numpy as np

relevance = [0.99, 0.94, 0.88, 0.74, 0.71, 0.68]
k = 5

from normalized_dcg import normalized_dcg
print(normalized_dcg(relevance, 5, 'standard'))