import numpy as np
from collections import deque
from typing import Any


class BaseFilter:
    """Interface for filters that process measurement arrays."""

    def reset(self) -> None:
        pass

    def update(self, measurement: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def copy(self) -> 'BaseFilter':
        return self.__class__(**self.__dict__)


class NoOpFilter(BaseFilter):
    def update(self, measurement: np.ndarray) -> np.ndarray:
        return measurement


class MovingAverageFilter(BaseFilter):
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        self.history = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self.history.clear()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        self.history.append(np.asarray(measurement))
        
        # history has shape (window_size, num_agents, dim)
        # Mean should be over axis 0
        return np.mean(np.array(self.history), axis=0)
    
    def copy(self) -> 'MovingAverageFilter':
        return MovingAverageFilter(self.window_size)