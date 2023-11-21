import itertools
from collections import defaultdict
from itertools import combinations
from operator import itemgetter
from typing import Dict
from typing import List
from typing import Tuple
import numpy as np


class SimilarItems:
    """_summary_

    Returns:
        _type_: _description_
    """
    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        def cals_similarity(a, b):
            """
            """
            return round(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)), 8)
        pairs = combinations(embeddings.keys(), 2)
        pair_sim = {pair: cals_similarity(
            embeddings[pair[0]], embeddings[pair[1]]) for pair in pairs}
        return pair_sim

    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        sim_dict = defaultdict(list)
        keys = list(set(itertools.chain.from_iterable(sim.keys())))
        for i in keys:
            for y in sim.keys():
                if i in y:
                    idx = 1 - y.index(i)
                    sim_dict[i].append((y[idx], sim[y]))

        for i in sim_dict:
            sim_dict[i] = list(itertools.islice(
                sorted(sim_dict[i], key=lambda tup: tup[1], reverse=True), top))

        return sim_dict

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = {}
        for i in knn_dict:
            weights = [1 + w[1] for w in knn_dict[i]]
            price = np.array(
                itemgetter(*[w[0] for w in knn_dict[i]])(prices))
            knn_price_dict[i] = np.round(np.sum(np.multiply(
                weights, price)) / np.sum(weights), 2)
        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        sim = SimilarItems.similarity(embeddings)
        sim_dict = SimilarItems.knn(sim, top)
        sim_price_dict = SimilarItems.knn_price(sim_dict, prices)
        return sim_price_dict
