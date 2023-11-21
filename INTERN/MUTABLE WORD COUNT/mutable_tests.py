import utils


def test_word_count() -> None:
    """Тест для корректной версии функции word_count"""
    texts = ["apple banana apple", "cherry orange", "apple"]
    count = utils.word_count(texts)
    assert count == {"apple": 3, "banana": 1, "cherry": 1, "orange": 1}


def test_word_count_tricky() -> None:
    """Тест для версии функции word_count с mutable объектом
    Этот тест должен вызвать AssertionError, так как версия
    функции с mutable объектом может давать неверные результаты.
    """
    texts2 = ["apple"]
    count = utils.word_count(texts2)
    assert count != {"apple": 4, "banana": 1, "cherry": 1, "orange": 1}
