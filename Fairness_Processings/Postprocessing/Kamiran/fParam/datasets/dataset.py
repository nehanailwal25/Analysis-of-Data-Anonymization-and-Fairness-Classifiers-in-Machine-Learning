from abc import ABC, abstractmethod
import copy


class Dataset(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        self.metadata = kwargs.pop('metadata', dict()) or dict()
        self.metadata.update({
            'transformer': '{}.__init__'.format(type(self).__name__),
            'params': kwargs,
            'previous': []
        })
        self.validate_dataset()

    def validate_dataset(self):
        """Error checking and type validation."""
        pass

    #Convenience method to return a copy of this dataset
    def copy(self, deepcopy=False):

        cpy = copy.deepcopy(self) if deepcopy else copy.copy(self)
        # preserve any user-created fields
        cpy.metadata = cpy.metadata.copy()
        cpy.metadata.update({
            'transformer': '{}.copy'.format(type(self).__name__),
            'params': {'deepcopy': deepcopy},
            'previous': [self]
        })
        return cpy

    @abstractmethod
    def export_dataset(self):
        """Save this Dataset to disk."""
        raise NotImplementedError

    @abstractmethod
    def split(self, num_or_size_splits, shuffle=False):
        raise NotImplementedError
