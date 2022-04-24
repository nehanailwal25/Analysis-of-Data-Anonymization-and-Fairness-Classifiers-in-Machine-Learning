from abc import abstractmethod
from collections import Hashable
from functools import wraps

from fParam.datasets import Dataset
from fParam.decorating_metaclass import ApplyDecorator


def _make_key(args, kwargs, unhashable, kwd_mark=(object(),)):
    """Simplified version of functools."""
    key = args
    if kwargs:
        key += kwd_mark
        for item in kwargs.items():
            if not isinstance(item[1], Hashable):
                return unhashable
            key += item
    return key

def memoize(func):
    """A little inefficient but we're just storing floats. """
    sentinal = object()
    unhashable = object()
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = _make_key(args, kwargs, unhashable)
        if key is unhashable:
            return func(*args, **kwargs)
        result = cache.get(key, sentinal)
        if result is not sentinal:
            return result
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper


BaseClass = ApplyDecorator(memoize)

class Metric(BaseClass):
    """Base class for metrics."""
    @abstractmethod
    def __init__(self, dataset):
        """Initialize a `Metrics` object. """
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError("dataset must be of Dataset class")
