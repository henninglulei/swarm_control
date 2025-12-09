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


# This MA filter is equivalent to the MLE when the noise is constant
class MovingAverageFilter(BaseFilter):
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        self.history = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self.history.clear()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        self.history.append(measurement)
        
        # history has shape (window_size, num_agents, dim)
        # Mean should be over axis 0
        return np.mean(np.array(self.history), axis=0)
    
    def copy(self) -> 'MovingAverageFilter':
        return MovingAverageFilter(self.window_size)
    

# RLS filter with forgetting factor < 1
class RecursiveLeastSquaresFilter(BaseFilter):
    def __init__(self, forgetting_factor: float, window_size: int, alpha: float, noise_covariance: np.ndarray):
        self.dimension = noise_covariance.shape[0]
        
        self.forgetting_factor = forgetting_factor
        self.window_size = window_size
        
        self.measurement_history = deque(maxlen=self.window_size)
        self.filtered_history = deque(maxlen=self.window_size)
        
        self.noise_covariance = noise_covariance
        self.alpha = alpha
        self.uncertainty_covariance = None
        
    def reset(self) -> None:
        self.measurement_history.clear()
        self.filtered_history.clear()
        self.uncertainty_covariance = None
        
    def update(self, measurement: np.ndarray) -> np.ndarray:
        self.measurement_history.append(measurement)
        
        if self.uncertainty_covariance is None:
            # This is the first incoming measurement
            self.filtered_history.append(np.zeros_like(measurement))
            self.uncertainty_covariance = np.tile(np.eye(self.dimension) * self.alpha, (measurement.shape[0], 1, 1))
        
        filtered_measurement = np.zeros_like(measurement)
        
        for i in range(measurement.shape[0]):        
            innovation_covariance = self.uncertainty_covariance[i] + self.noise_covariance
            gain = self.uncertainty_covariance[i] @ np.linalg.inv(innovation_covariance)
            
            innovation = measurement[i] - self.filtered_history[-1][i]
            filtered_measurement[i] = self.filtered_history[-1][i] + gain @ innovation
            
            self.uncertainty_covariance[i] = 1 / self.forgetting_factor * ((np.eye(self.dimension) - gain) @ self.uncertainty_covariance[i])
            
        self.filtered_history.append(filtered_measurement)
        
        return filtered_measurement
    
    def copy(self) -> 'RecursiveLeastSquaresFilter':
        return RecursiveLeastSquaresFilter(
            forgetting_factor=self.forgetting_factor,
            window_size=self.window_size,
            alpha=self.alpha,
            noise_covariance=self.noise_covariance
        )