from typing import Callable


def memoize(func: Callable) -> Callable:
    """Memoize function"""
    cache = {}

    def wrapper(*args, **kwargs):
        # Создаем уникальный ключ на основе аргументов и их значений
        key = hash(f"{str(args)}{str(kwargs)}")

        # Если результат для данного ключа уже кэширован, возвращаем его
        if key in cache:
            return cache[key]

        # В противном случае, вызываем функцию и кэшируем ее результат
        result = func(*args, **kwargs)
        cache[key] = result

        return result

    return wrapper


