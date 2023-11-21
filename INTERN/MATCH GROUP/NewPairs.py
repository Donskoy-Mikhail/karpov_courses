from typing import List
from typing import Tuple
from collections import defaultdict


def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """extend_matches
    """
    d = defaultdict(list)
    pairs = [tuple(sorted(i)) for i in pairs]
    for i in pairs:
        for el in i:
            d[el] += [e for e in i if e != el]
    for i in d:
        for y in d[i]:
            if y in d:
                d[i] += d[y]
                d[i] = list(set(d[i]))
        if i not in d[i]:
            d[i].append(i)

    pairs = [tuple(sorted(v)) for _, v in d.items()]
    return sorted(list(set(pairs)))

# p = [(1, 2), (7, 2)]
p = [(4, 5), (4, 8), (1, 2), (7, 2)]
exp = [(1, 2), (7, 2)]

print(extend_matches(exp))
print(extend_matches(p))