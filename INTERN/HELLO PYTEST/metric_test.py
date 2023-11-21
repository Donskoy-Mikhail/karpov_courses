import metrics


def test_profit() -> None:
    """test_profit"""
    assert metrics.profit([1, 2, 3], [1, 2, 3]) == 0
    assert metrics.profit([1, 2, 3], [1, 1, 1]) == 3
    assert metrics.profit([1, 1, 1], [1, 2, 3]) == -3


def test_margin() -> None:
    """test_margin"""
    assert metrics.margin([1, 2, 3], [1, 2, 3]) == 0.0

    assert metrics.margin([10, 20, 30], [5, 10, 15]) == 0.5

    assert metrics.margin([5, 10, 15], [10, 20, 30]) == -1

def test_markup() -> None:
    """test_markup"""
    assert metrics.markup([1, 2, 3], [1, 2, 3]) == 0.0

    assert metrics.markup([10, 20, 30], [5, 10, 15]) == 1.0

    assert metrics.markup([5, 10, 15], [10, 20, 30]) == -0.5
