from abc import ABCMeta, abstractmethod

class BaseFilter(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, text: str) -> bool:
        pass