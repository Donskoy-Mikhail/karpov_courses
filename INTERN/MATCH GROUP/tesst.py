from typing import List
from typing import Tuple
from collections import defaultdict


def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """extend_matches
    """
    d = set(pairs)
    print(d)

    for i, par in enumerate(pairs):
       par = set(par)
       if i < len(pairs):
           for par2 in pairs[i:]:
               par2 = set(par2)
               if len(par & par2) != 0:
                   d.add(tuple(par ^ par2))
    d.remove(tuple())
    return d

# p = [(1, 2), (7, 2)]
p = [(4, 5), (4, 8), (1, 2), (7, 2)]
exp = [(1, 2), (7, 2)]

print(extend_matches(exp))