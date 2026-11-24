"""
Composite Anomaly Generator.

Orchestrates multiple physics-based models to create realistic
failure scenarios and mixed-mode failures.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import yaml

from .base_generator import BaseAnomalyModel
from .physics_models import (
    ParisCrackModel,
    ArrheniusThermalModel,
    BearingWearModel,
    ImbalanceModel,
    MechanicalLoosenessModel
)


class PhysicsAnomalyGenerator:
    """
    High-level generator that combines physics-based anomaly models.
    
    Supports:
    - Single failure mode injection
    - Sequential failure cascade (e.g., imbalance -> bearing wear -> crack)
    - Concurrent multiple failures
    - Configurable severity and timing
    
    Usage:
        generator = PhysicsAnomalyGenerator(sensor_name='BA5_Left Motor')
        df_anomaly, labels = generator.generate_scenario(
            base_df, 
            scenario='bearing_wear',
            severity=1.5
        )
    """
    
    def __init__(self, 
                 sensor_name: str = 'BA5_Left Motor',
                 random_state: Optional[int] = None):
        """
        Initialize composite generator.
        
        Args:
            sensor_name: Target sensor
            random_state: Random seed for reproducibility
        """
        self.sensor_name = sensor_name
        self.random_state = random_state
        
        # Initialize all available models
        self.models = {
            'crack': ParisCrackModel(sensor_name, random_state),
            'thermal': ArrheniusThermalModel(sensor_name, random_state),
            'bearing_wear': BearingWearModel(sensor_name, random_state),            'mechanical_looseness': MechanicalLoosenessModel(sensor_name, random_state),            'imbalance': ImbalanceModel(sensor_name, random_state)
        }
    
    def generate_scenario(self,
                         base_df: pd.DataFrame,
                         scenario: str,
                         severity: float = 1.0,
                         **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate specific failure scenario.
        
        Args:
            base_df: Clean baseline data
            scenario: One of ['crack', 'thermal', 'bearing_wear', 'imbalance', 
                             'cascade', 'mixed']
            severity: Severity multiplier
            **kwargs: Model-specific parameters
            
        Returns:
            (anomalous_df, labels)
        """
        if scenario in self.models:
            # Single failure mode
            return self.models[scenario].inject(base_df, severity=severity, **kwargs)
        
        elif scenario == 'cascade':
            # Sequential failure progression
            return self._generate_cascade(base_df, severity, **kwargs)
        
        elif scenario == 'mixed':
            # Concurrent multiple failures
            return self._generate_mixed(base_df, severity, **kwargs)
        
        else:
            raise ValueError(f"Unknown scenario: {scenario}. "
                           f"Available: {list(self.models.keys()) + ['cascade', 'mixed']}")
    
    def _generate_cascade(self,
                         base_df: pd.DataFrame,
                         severity: float = 1.0,
                         **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate cascading failure sequence.
        
        Realistic progression:
        1. Imbalance (early, mild)
        2. Bearing wear (mid, moderate) - caused by imbalance
        3. Crack initiation (late, severe) - caused by cyclic stress
        
        Args:
            base_df: Clean data
            severity: Overall severity
            
        Returns:
            (cascaded_df, labels)
        """
        df = base_df.copy()
        n = len(df)
        labels = np.zeros(n, dtype=int)
        
        # Stage 1: Imbalance (0-50%)
        df, labels1 = self.models['imbalance'].inject(
            df,
            start_ratio=0.0,
            duration_ratio=0.5,
            severity=severity * 0.8
        )
        labels = np.maximum(labels, labels1)
        
        # Stage 2: Bearing wear (30-70%) - overlap with imbalance
        df, labels2 = self.models['bearing_wear'].inject(
            df,
            start_ratio=0.3,
            duration_ratio=0.4,
            severity=severity * 1.0
        )
        labels = np.maximum(labels, labels2)
        
        # Stage 3: Crack (60-100%) - final stage
        df, labels3 = self.models['crack'].inject(
            df,
            start_ratio=0.6,
            duration_ratio=0.4,
            severity=severity * 1.2
        )
        labels = np.maximum(labels, labels3)
        
        return df, labels
    
    def _generate_mixed(self,
                       base_df: pd.DataFrame,
                       severity: float = 1.0,
                       components: Optional[List[str]] = None,
                       **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate concurrent multiple failures.
        
        Args:
            base_df: Clean data
            severity: Overall severity
            components: List of failure modes to mix (default: random 2-3)
            
        Returns:
            (mixed_df, labels)
        """
        df = base_df.copy()
        n = len(df)
        labels = np.zeros(n, dtype=int)
        
        # Select components if not specified
        if components is None:
            rng = np.random.default_rng(self.random_state)
            n_components = rng.integers(2, 4)
            components = list(rng.choice(
                list(self.models.keys()), 
                size=n_components, 
                replace=False
            ))
        
        # Inject each component with random timing
        rng = np.random.default_rng(self.random_state)
        
        for component in components:
            # Randomize start and duration
            start_ratio = rng.uniform(0.0, 0.4)
            duration_ratio = rng.uniform(0.3, 0.8)
            component_severity = severity * rng.uniform(0.8, 1.2)
            
            df, labels_i = self.models[component].inject(
                df,
                start_ratio=start_ratio,
                duration_ratio=duration_ratio,
                severity=component_severity
            )
            
            # Union of all anomalies
            labels = np.maximum(labels, labels_i)
        
        return df, labels
    
    def generate_batch(self,
                      base_df: pd.DataFrame,
                      output_dir: Union[str, Path],
                      scenarios: Optional[List[str]] = None,
                      severities: Optional[List[float]] = None,
                      target_anomaly_ratios: Optional[List[float]] = None,
                      prefix: str = 'physics_test') -> Dict:
        """
        Generate multiple test scenarios and save to files.
        
        Args:
            base_df: Clean baseline data
            output_dir: Output directory
            scenarios: List of scenarios (default: all single modes + cascade + mixed)
            severities: List of severity levels (default: [1.0])
            target_anomaly_ratios: List of target anomaly ratios (0-1). If provided,
                                   will adjust duration_ratio to achieve target.
            prefix: Filename prefix
            
        Returns:
            Dictionary with generation summary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default scenarios
        if scenarios is None:
            scenarios = list(self.models.keys()) + ['cascade', 'mixed']
        
        # Default severities
        if severities is None:
            severities = [1.0]
        
        # Default target ratios (will use model defaults if None)
        if target_anomaly_ratios is None:
            target_anomaly_ratios = [None]  # Use model defaults
        
        results = []
        
        for scenario in scenarios:
            for severity in severities:
                for target_ratio in target_anomaly_ratios:
                    # Calculate duration_ratio to achieve target_anomaly_ratio
                    # Simple heuristic: duration_ratio ≈ target_ratio (not perfect but reasonable)
                    if target_ratio is not None:
                        duration_ratio = min(0.9, max(0.1, target_ratio))
                        start_ratio = 1.0 - duration_ratio  # Start later for shorter duration
                    else:
                        duration_ratio = None
                        start_ratio = None
                    
                    # Generate
                    kwargs = {}
                    if duration_ratio is not None:
                        kwargs['duration_ratio'] = duration_ratio
                        kwargs['start_ratio'] = start_ratio
                    
                    df_anomaly, labels = self.generate_scenario(
                        base_df,
                        scenario=scenario,
                        severity=severity,
                        **kwargs
                    )
                    
                    # Create filename
                    sev_str = f"{severity:.1f}".replace('.', '_')
                    if target_ratio is not None:
                        ratio_str = f"_r{int(target_ratio*100)}"
                    else:
                        ratio_str = ""
                    filename = f"{prefix}_{scenario}_sev{sev_str}{ratio_str}"
                    
                    # Save data
                    data_path = output_dir / f"{filename}_data.csv"
                    label_path = output_dir / f"{filename}_labels.csv"
                    
                    df_anomaly.to_csv(data_path, index=False)
                    pd.DataFrame({'label': labels}).to_csv(label_path, index=False)
                    
                    # Summary
                    anomaly_count = labels.sum()
                    anomaly_ratio = anomaly_count / len(labels)
                    
                    results.append({
                        'scenario': scenario,
                        'severity': severity,
                        'target_ratio': target_ratio,
                        'filename': filename,
                        'n_samples': len(labels),
                        'n_anomalies': int(anomaly_count),
                        'anomaly_ratio': f"{anomaly_ratio:.2%}"
                    })
                
                print(f"Generated: {filename}")
                print(f"  Anomalies: {anomaly_count} / {len(labels)} ({anomaly_ratio:.2%})")
        
        # Save summary
        summary_path = output_dir / f"{prefix}_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump({
                'sensor': self.sensor_name,
                'base_file_length': len(base_df),
                'scenarios': results
            }, f)
        
        print(f"\nGeneration complete. Summary saved to {summary_path}")
        
        return {
            'output_dir': str(output_dir),
            'n_scenarios': len(results),
            'scenarios': results
        }
    
    def get_model_info(self) -> Dict:
        """Get information about all available models."""
        return {
            name: {
                'class': model.__class__.__name__,
                'description': model.__class__.__doc__.split('\n')[1].strip()
            }
            for name, model in self.models.items()
        }


def demonstrate_models(sensor_name: str = 'BA5_Left Motor',
                      data_file: str = 'data/example_preprocessed.csv',
                      output_dir: str = 'data_physics_demo'):
    """
    Demonstration function to generate examples of all physics models.
    
    Args:
        sensor_name: Target sensor
        data_file: Path to clean baseline data
        output_dir: Output directory for generated data
    """
    import pandas as pd
    
    print("="*60)
    print("PHYSICS-BASED ANOMALY GENERATION DEMO")
    print("="*60)
    
    # Load baseline data
    print(f"\nLoading baseline data: {data_file}")
    df_base = pd.read_csv(data_file)
    print(f"  Samples: {len(df_base)}")
    print(f"  Sensor: {sensor_name}")
    
    # Initialize generator
    generator = PhysicsAnomalyGenerator(sensor_name=sensor_name, random_state=42)
    
    # Show available models
    print("\nAvailable Physics Models:")
    for name, info in generator.get_model_info().items():
        print(f"  - {name}: {info['description']}")
    
    # Generate batch
    print(f"\nGenerating test scenarios...")
    summary = generator.generate_batch(
        df_base,
        output_dir=output_dir,
        scenarios=['crack', 'thermal', 'bearing_wear', 'imbalance', 'cascade', 'mixed'],
        severities=[1.0, 1.5],
        prefix='physics_demo'
    )
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"Generated {summary['n_scenarios']} scenarios")
    print(f"Output directory: {summary['output_dir']}")
    
    return summary


if __name__ == "__main__":
    # Run demonstration
    demonstrate_models()
