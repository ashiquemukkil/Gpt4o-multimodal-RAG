from abc import ABC, abstractmethod

class MultiModalDB(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def add(self):
        pass