import metrics


def test_non_int_clicks():
    """
    Test that ctr function raises a TypeError when clicks is not an integer.
    """
    try:
        metrics.ctr(1.5, 2)
    except TypeError:
        pass
    else:
        raise AssertionError("Non int clicks not handled")


def test_non_int_views():
    """
    Test that ctr function raises a TypeError when views is not an integer.
    """
    try:
        metrics.ctr(1, "2")
    except TypeError:
        pass
    else:
        raise AssertionError("Non int views not handled")


def test_non_positive_clicks():
    """
    Test that ctr function raises a ValueError when clicks is not positive.
    """
    try:
        metrics.ctr(-1, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Non positive clicks not handled")


def test_non_positive_views():
    """
    Test that ctr function raises a ValueError when views is not positive.
    """
    try:
        metrics.ctr(1, -2)
    except ValueError:
        pass
    else:
        raise AssertionError("Non positive views not handled")


def test_clicks_greater_than_views():
    """
    Test that ctr function raises a ValueError when clicks is greater than views.
    """
    try:
        metrics.ctr(3, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Clicks greater than views not handled")


def test_zero_views():
    """
    Test that ctr function raises a ZeroDivisionError when views is zero.
    """
    try:
        metrics.ctr(0, 0)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("Zero views not handled")
