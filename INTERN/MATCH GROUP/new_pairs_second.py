from typing import List
from typing import Tuple


def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """extend_matches
    """
    set_of_pairs = set(pairs)
    for i, par in enumerate(pairs):
        par = set(par)
        if i < len(pairs):
            for par2 in pairs[i:]:
                par2 = set(par2)
                if len(par & par2) != 0:
                    if tuple(par ^ par2) not in set_of_pairs:
                        set_of_pairs.add(tuple(par ^ par2))
                        pairs.append(tuple(par ^ par2))
    set_of_pairs.discard(tuple())
    set_of_pairs = list(set_of_pairs)
    set_of_pairs = list(set([tuple(sorted(i)) for i in set_of_pairs]))
    set_of_pairs = sorted(set_of_pairs)
    return set_of_pairs
