"""
Base class for physics-based anomaly models.

Defines the interface and common functionality for all failure models.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional


class BaseAnomalyModel(ABC):
    """
    Abstract base class for physics-based anomaly injection.
    
    All failure models should inherit from this class and implement
    the `inject` method with specific physics-based degradation logic.
    
    Attributes:
        name (str): Human-readable model name
        sensor_name (str): Target sensor (e.g., 'BA5_Left Motor')
        random_state (int): Random seed for reproducibility
        rng (np.random.Generator): Random number generator
    """
    
    def __init__(self, 
                 name: str,
                 sensor_name: str = 'BA5_Left Motor',
                 random_state: Optional[int] = None):
        """
        Initialize base anomaly model.
        
        Args:
            name: Model identifier
            sensor_name: Target sensor name
            random_state: Random seed (None for random)
        """
        self.name = name
        self.sensor_name = sensor_name
        self.random_state = random_state
        
        # Initialize random generator
        if random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()
        
        # Column names for target sensor
        self.col_x = f"{sensor_name}_X"
        self.col_y = f"{sensor_name}_Y"
        self.col_z = f"{sensor_name}_Z"
        self.col_t = f"{sensor_name}_T"
        self.target_cols = [self.col_x, self.col_y, self.col_z, self.col_t]
        
        # Track injection metadata
        self.injection_log = []
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        missing = [col for col in self.target_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {self.sensor_name}: {missing}")
    
    def compute_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for target columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with mean, std, min, max for each column
        """
        stats = {}
        for col in self.target_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        return stats
    
    @abstractmethod
    def inject(self, 
               df: pd.DataFrame,
               start_ratio: float = 0.5,
               duration_ratio: float = 0.3,
               severity: float = 1.0,
               **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Inject anomalies into the data.
        
        Args:
            df: Input DataFrame (will be copied, not modified in place)
            start_ratio: Where to start injection (0-1, fraction of data length)
            duration_ratio: How long to inject (0-1, fraction of data length)
            severity: Severity multiplier (1.0 = normal, higher = more severe)
            **kwargs: Model-specific parameters
            
        Returns:
            Tuple of (modified DataFrame, labels array)
            labels: 0 = normal, 1 = anomaly
        """
        pass
    
    def add_noise(self, 
                  values: np.ndarray, 
                  noise_level: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to values for realism.
        
        Args:
            values: Input array
            noise_level: Standard deviation of noise (relative to value range)
            
        Returns:
            Values with added noise
        """
        noise = self.rng.normal(0, noise_level, size=values.shape)
        return values + noise
    
    def smooth_transition(self,
                          n_points: int,
                          transition_ratio: float = 0.15) -> np.ndarray:
        """
        Generate smooth transition curve (sigmoid-like).
        
        This avoids sharp boundaries between normal and anomalous regions.
        
        Args:
            n_points: Number of points in transition
            transition_ratio: Fraction of points for ramp-up/down
            
        Returns:
            Array of transition weights [0, 1]
        """
        transition_len = int(n_points * transition_ratio)
        if transition_len < 2:
            transition_len = 2

        # Sigmoid for smooth transition
        x = np.linspace(-6, 6, transition_len)
        sigmoid = 1 / (1 + np.exp(-x))

        # Full transition curve
        curve = np.zeros(n_points)

        # Guard: if n_points is too small for the transition, clamp
        actual_ramp = min(transition_len, n_points)
        curve[:actual_ramp] = sigmoid[-actual_ramp:]
        
        # Full effect
        if n_points > 2 * transition_len:
            curve[transition_len:-transition_len] = 1.0
        
        # Ramp down (optional, for transient faults)
        # curve[-transition_len:] = 1 - sigmoid
        # For permanent faults, keep high:
        curve[-actual_ramp:] = 1.0
        
        return curve
    
    def get_injection_summary(self) -> Dict:
        """Get summary of injections performed."""
        return {
            'model': self.name,
            'sensor': self.sensor_name,
            'injections': self.injection_log
        }
