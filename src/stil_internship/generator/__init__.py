import abc


class Generator(abc.ABC):
    def __init__(self):
        ...

    @property
    @abc.abstractmethod
    def name(self):
        ...

    @abc.abstractmethod
    def gen(self, m, n):
        ...


