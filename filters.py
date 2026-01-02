import numpy as np
from collections import deque
from typing import Any


class BaseFilter:
    """Interface for filters that process measurement arrays."""

    def reset(self) -> None:
        pass

    def update(self, T_measurements: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def copy(self) -> 'BaseFilter':
        return self.__class__(**self.__dict__)


class NoOpFilter(BaseFilter):
    def update(self, T_measurements: list) -> np.ndarray:
        # T_measurements is a list of T arrays, each shape (num_agents, dim)
        # For NoOpFilter, just return the first measurement
        return T_measurements[0]
    
    
# Maximum Likelihood Estimator based on the T independent measurements
class MaximumLikelihoodEstimator(BaseFilter):
    def update(self, T_measurements: list) -> np.ndarray:
        # T_measurements is a list of T arrays, each shape (num_agents, dim)
        # Average across T measurements
        return np.mean(np.array(T_measurements), axis=0)
    

class RecursiveLeastSquaresFilter(BaseFilter):
    def __init__(self, forgetting_factor: float, alpha: float, noise_covariance: np.ndarray):
        self.dimension = noise_covariance.shape[0]
        
        self.forgetting_factor = forgetting_factor
        
        self.last_estimate = None
        
        self.noise_covariance = noise_covariance
        self.alpha = alpha
        self.uncertainty_covariance = None
        
    def reset(self) -> None:
        self.last_estimate = None
        self.uncertainty_covariance = None
        
    def update(self, T_measurements: list) -> np.ndarray:
        # T_measurements is a list of T arrays, each shape (num_agents, dim)
        # Average across T measurements first, very simple use of these. Might improve later
        measurement = np.mean(np.array(T_measurements), axis=0)  # shape (num_agents, dim)
        if self.uncertainty_covariance is None:
            # This is the first incoming measurement
            self.noise_covariance = self.noise_covariance / len(T_measurements)
            self.last_estimate = measurement.copy()
            self.uncertainty_covariance = np.tile(np.eye(self.dimension) * self.alpha, (measurement.shape[0], 1, 1))
            return self.last_estimate   
            
        filtered_measurement = np.zeros_like(measurement)
        
        for i in range(measurement.shape[0]): 
            self.uncertainty_covariance[i] = (1.0 / self.forgetting_factor) * self.uncertainty_covariance[i]
                 
            innovation_covariance = self.uncertainty_covariance[i] + self.noise_covariance
            gain = self.uncertainty_covariance[i] @ np.linalg.inv(innovation_covariance)
            
            innovation = measurement[i] - self.last_estimate[i]
            filtered_measurement[i] = self.last_estimate[i] + gain @ innovation
            
            self.uncertainty_covariance[i] = (np.eye(self.dimension) - gain) @ self.uncertainty_covariance[i]
            
        self.last_estimate = filtered_measurement
        
        return filtered_measurement
    
    def copy(self) -> 'RecursiveLeastSquaresFilter':
        return RecursiveLeastSquaresFilter(
            forgetting_factor=self.forgetting_factor,
            alpha=self.alpha,
            noise_covariance=self.noise_covariance
        )