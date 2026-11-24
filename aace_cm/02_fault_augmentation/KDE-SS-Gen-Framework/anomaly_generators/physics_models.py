"""
Physics-based Anomaly Models for Industrial Motor Failures.

Implements realistic failure progression models based on established
theories from mechanical engineering and materials science.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from .base_generator import BaseAnomalyModel


class ParisCrackModel(BaseAnomalyModel):
    """
    Fatigue crack growth model based on Paris' Law.
    
    Paris' Law: da/dN = C * (ΔK)^m
    where:
    - da/dN: crack growth rate per cycle
    - ΔK: stress intensity factor range
    - C, m: material constants (m typically 2-4)
    
    This models progressive vibration increase due to crack propagation
    in structural components (shaft, housing, etc.).
    
    Physical manifestation:
    - Exponential growth in vibration amplitude
    - Affects all axes (X, Y, Z) but potentially asymmetric
    - Temperature may increase slightly due to friction
    """
    
    def __init__(self, 
                 sensor_name: str = 'BA5_Left Motor',
                 random_state: Optional[int] = None):
        super().__init__(name='Paris_Crack', 
                        sensor_name=sensor_name,
                        random_state=random_state)
        
        # Paris law constants (typical for steel)
        self.C = 1e-12  # Crack growth coefficient
        self.m = 3.0    # Paris exponent
    
    def inject(self,
               df: pd.DataFrame,
               start_ratio: float = 0.3,
               duration_ratio: float = 0.5,
               severity: float = 1.0,
               initial_crack_size: float = 0.1,
               critical_crack_size: float = 5.0,
               **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Inject crack growth progression.

        Args:
            df: Input DataFrame
            start_ratio: When crack becomes detectable (0-1)
            duration_ratio: Crack propagation period (0-1)
            severity: Controls final vibration amplification (0-1).
                       Maps to a sigmoid amplification factor:
                       sev=0.1 → ~1.1x  (barely perceptible)
                       sev=0.5 → ~1.5x  (moderate crack)
                       sev=1.0 → ~2.0x  (maximum amplification)
            initial_crack_size: Starting crack size (mm) — not used in amplification calc
            critical_crack_size: Size at failure (mm) — not used in amplification calc

        Returns:
            (modified_df, labels)
        """
        self.validate_data(df)
        df = df.copy()
        n_samples = len(df)
        labels = np.zeros(n_samples, dtype=int)

        start_idx = int(n_samples * start_ratio)
        duration = int(n_samples * duration_ratio)
        end_idx = min(start_idx + duration, n_samples)

        if end_idx <= start_idx:
            return df, labels

        n_cycles = end_idx - start_idx

        # ── Severity → amplification via sigmoid ───────────────────────────────────
        # Keeps amplification in [1.0, 2.0] so severity=1 never produces extreme values.
        sev_c = max(0.01, min(1.0, severity))
        A = 9.0   # steepness of sigmoid
        amp_max = 2.0
        final_amp = 1.0 + (amp_max - 1.0) / (1.0 + np.exp(-A * (sev_c - 0.5)))

        # Smooth ramp from 1.0 to final_amp during the fault window
        t = np.linspace(0, 1, n_cycles)
        ramp = 1.0 / (1.0 + np.exp(-8 * (t - 0.4)))  # ramps at t≈40%
        amplification_factors = 1.0 + (final_amp - 1.0) * ramp

        transition = self.smooth_transition(n_cycles, transition_ratio=0.1)
        amplification_factors = 1.0 + (amplification_factors - 1.0) * transition

        stats = self.compute_stats(df)

        for idx, amp in enumerate(amplification_factors):
            row_idx = start_idx + idx
            axis_weights = [1.0, 0.7, 0.5]

            for col, weight in zip([self.col_x, self.col_y, self.col_z], axis_weights):
                df.at[row_idx, col] = df.at[row_idx, col] * (1 + (amp - 1) * weight)

            df.at[row_idx, self.col_t] += (amp - 1) * stats[self.col_t]['std'] * 0.3
            labels[row_idx] = 1

        for col in [self.col_x, self.col_y, self.col_z]:
            noise = self.rng.normal(0, stats[col]['std'] * 0.05, size=n_cycles)
            df.loc[start_idx:end_idx-1, col] += noise

        self.injection_log.append({
            'type': 'crack_growth',
            'start': start_idx,
            'end': end_idx,
            'severity': severity,
            'final_amp': final_amp,
        })

        return df, labels


class ArrheniusThermalModel(BaseAnomalyModel):
    """
    Thermal degradation model based on Arrhenius equation.
    
    Arrhenius: k = A * exp(-Ea / (R*T))
    where:
    - k: degradation rate
    - Ea: activation energy
    - R: gas constant
    - T: absolute temperature
    
    Models thermal aging/overheating scenarios:
    - Bearing lubrication breakdown
    - Insulation degradation
    - Thermal expansion effects
    
    Physical manifestation:
    - Gradual temperature increase (monotonic drift)
    - Secondary vibration increase due to thermal expansion
    - Accelerating degradation (positive feedback)
    """
    
    def __init__(self,
                 sensor_name: str = 'BA5_Left Motor',
                 random_state: Optional[int] = None):
        super().__init__(name='Arrhenius_Thermal',
                        sensor_name=sensor_name,
                        random_state=random_state)
        
        # Arrhenius parameters
        self.Ea = 50000  # Activation energy (J/mol)
        self.R = 8.314   # Gas constant
        self.A = 1e10    # Pre-exponential factor
    
    def inject(self,
               df: pd.DataFrame,
               start_ratio: float = 0.2,
               duration_ratio: float = 0.6,
               severity: float = 1.0,
               temp_increase_rate: float = 0.5,
               **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Inject thermal degradation.
        
        Args:
            df: Input DataFrame
            start_ratio: When overheating begins
            duration_ratio: Duration of thermal exposure
            severity: Multiplier for degradation rate
            temp_increase_rate: °C per timestep
            
        Returns:
            (modified_df, labels)
        """
        self.validate_data(df)
        df = df.copy()
        n_samples = len(df)
        labels = np.zeros(n_samples, dtype=int)
        
        start_idx = int(n_samples * start_ratio)
        duration = int(n_samples * duration_ratio)
        end_idx = min(start_idx + duration, n_samples)
        
        if end_idx <= start_idx:
            return df, labels
        
        n_points = end_idx - start_idx
        stats = self.compute_stats(df)
        
        # Temperature drift with Arrhenius acceleration
        base_temp = df[self.col_t].iloc[start_idx:end_idx].mean()
        time_steps = np.arange(n_points)
        
        # Non-linear temperature increase (accelerating)
        temp_drift = temp_increase_rate * severity * (time_steps ** 1.2)
        
        # Apply smooth transition
        transition = self.smooth_transition(n_points, transition_ratio=0.15)
        temp_drift = temp_drift * transition
        
        # Calculate degradation factor from temperature
        temps_K = base_temp + 273.15 + temp_drift
        degradation_rates = self.A * np.exp(-self.Ea / (self.R * temps_K))
        degradation_rates = degradation_rates / degradation_rates[0]  # Normalize
        
        # Apply effects
        for idx, (temp_inc, deg_factor) in enumerate(zip(temp_drift, degradation_rates)):
            row_idx = start_idx + idx
            
            # Temperature increase
            df.at[row_idx, self.col_t] += temp_inc
            
            # Thermal expansion causes vibration increase
            # Higher temps -> more vibration
            vib_increase = (deg_factor - 1) * 0.3
            
            for col in [self.col_x, self.col_y, self.col_z]:
                base_val = df.at[row_idx, col]
                df.at[row_idx, col] = base_val * (1 + vib_increase)
            
            # CRITICAL: Once thermal degradation starts, it's continuous (doesn't cool down)
            # Label all points in the degradation period as anomaly
            labels[row_idx] = 1
        
        # Add thermal noise
        thermal_noise = self.rng.normal(0, stats[self.col_t]['std'] * 0.1, size=n_points)
        df.loc[start_idx:end_idx-1, self.col_t] += thermal_noise
        
        self.injection_log.append({
            'type': 'thermal_degradation',
            'start': start_idx,
            'end': end_idx,
            'severity': severity,
            'max_temp_increase': temp_drift[-1]
        })
        
        return df, labels


class BearingWearModel(BaseAnomalyModel):
    """
    Bearing wear progression model.
    
    Based on Archard's wear equation:
    V = k * (W * s) / H
    where:
    - V: wear volume
    - k: wear coefficient
    - W: normal load
    - s: sliding distance
    - H: hardness
    
    Models bearing surface degradation:
    - Progressive pitting/spalling
    - Increased clearance
    - Periodic impact signatures
    
    Physical manifestation:
    - Increasing vibration with periodic spikes
    - Specific frequency components (bearing defect frequencies)
    - Slight temperature increase from friction
    """
    
    def __init__(self,
                 sensor_name: str = 'BA5_Left Motor',
                 random_state: Optional[int] = None):
        super().__init__(name='Bearing_Wear',
                        sensor_name=sensor_name,
                        random_state=random_state)
        
        # Wear parameters
        self.k_wear = 1e-6  # Wear coefficient
        self.bearing_freq = 50  # Defect frequency (Hz) - simplified
    
    def inject(self,
               df: pd.DataFrame,
               start_ratio: float = 0.4,
               duration_ratio: float = 0.4,
               severity: float = 1.0,
               spike_probability: float = 0.15,
               **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Inject bearing wear pattern.
        
        Args:
            df: Input DataFrame
            start_ratio: When wear becomes detectable
            duration_ratio: Wear progression period
            severity: Wear rate multiplier
            spike_probability: Probability of impact spike per timestep
            
        Returns:
            (modified_df, labels)
        """
        self.validate_data(df)
        df = df.copy()
        n_samples = len(df)
        labels = np.zeros(n_samples, dtype=int)
        
        start_idx = int(n_samples * start_ratio)
        duration = int(n_samples * duration_ratio)
        end_idx = min(start_idx + duration, n_samples)
        
        if end_idx <= start_idx:
            return df, labels
        
        n_points = end_idx - start_idx
        stats = self.compute_stats(df)
        
        # Progressive wear: baseline vibration increases
        wear_progression = np.linspace(0, 1, n_points) ** 1.5  # Non-linear wear
        baseline_increase = wear_progression * severity * 0.5
        
        # Apply smooth transition
        transition = self.smooth_transition(n_points, transition_ratio=0.2)
        baseline_increase = baseline_increase * transition
        
        # Generate periodic impact spikes
        spike_mask = self.rng.random(n_points) < (spike_probability * severity)
        spike_magnitudes = self.rng.exponential(2.0, size=n_points) * spike_mask
        
        # Apply effects
        for idx in range(n_points):
            row_idx = start_idx + idx
            
            # Baseline wear effect
            for col in [self.col_x, self.col_y, self.col_z]:
                base_val = df.at[row_idx, col]
                # Gradual increase
                df.at[row_idx, col] = base_val * (1 + baseline_increase[idx])
                
                # Add impact spikes
                if spike_mask[idx]:
                    spike_val = stats[col]['std'] * spike_magnitudes[idx]
                    df.at[row_idx, col] += spike_val
            
            # Friction heating
            temp_increase = baseline_increase[idx] * stats[self.col_t]['std'] * 0.4
            df.at[row_idx, self.col_t] += temp_increase
            
            # CRITICAL: Once bearing wear begins, it's permanent (doesn't heal)
            # Label all points from start as anomaly
            labels[row_idx] = 1
        
        self.injection_log.append({
            'type': 'bearing_wear',
            'start': start_idx,
            'end': end_idx,
            'severity': severity,
            'spike_count': spike_mask.sum()
        })
        
        return df, labels


class ImbalanceModel(BaseAnomalyModel):
    """
    Rotor imbalance model.
    
    Based on centrifugal force: F = m * e * ω²
    where:
    - m: unbalanced mass
    - e: eccentricity
    - ω: angular velocity
    
    Models mass distribution problems:
    - Rotor imbalance (manufacturing, wear, deposits)
    - Misalignment
    - Bent shaft
    
    Physical manifestation:
    - Synchronous vibration at 1x RPM
    - Predominantly radial (X, Y), minimal axial (Z)
    - Amplitude proportional to speed²
    - Phase relationship between axes
    """
    
    def __init__(self,
                 sensor_name: str = 'BA5_Left Motor',
                 random_state: Optional[int] = None):
        super().__init__(name='Rotor_Imbalance',
                        sensor_name=sensor_name,
                        random_state=random_state)
        
        # Assume typical motor speed
        self.rpm = 1800  # RPM
        self.omega = 2 * np.pi * self.rpm / 60  # rad/s
    
    def inject(self,
               df: pd.DataFrame,
               start_ratio: float = 0.0,
               duration_ratio: float = 1.0,
               severity: float = 1.0,
               imbalance_mass: float = 0.01,  # kg
               eccentricity: float = 0.005,   # m
               **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Inject rotor imbalance.

        The dominant vibration signature of imbalance is a 1x RPM sinusoidal component.
        With 1 Hz sampling and ~30 Hz rotor frequency, the sinusoid aliases to near-DC
        (one full phase wrap every ~2 samples), which blends into the baseline.
        We therefore model the effect as a direct scaling of the baseline RMS:
          vib_scale = 1 + force_magnitude
        where force_magnitude = m · e · omega^2 · severity.
        The scale factor is directly proportional to severity, giving a linear
        delta-RMS progression across severity levels.

        Args:
            df: Input DataFrame
            start_ratio: When imbalance appears
            duration_ratio: How long imbalance lasts
            severity: Directly scales vibration amplitude.  severity=0.1 → ~+7% RMS,
                       severity=1.0 → ~+71% RMS (ISO Zone C/D).
            imbalance_mass: Unbalanced mass (kg)
            eccentricity: Distance from rotation axis (m)

        Returns:
            (modified_df, labels)
        """
        self.validate_data(df)
        df = df.copy()
        n_samples = len(df)
        labels = np.zeros(n_samples, dtype=int)

        start_idx = int(n_samples * start_ratio)
        duration = int(n_samples * duration_ratio)
        end_idx = min(start_idx + duration, n_samples)

        if end_idx <= start_idx:
            return df, labels

        n_points = end_idx - start_idx
        stats = self.compute_stats(df)

        # Centrifugal force magnitude
        force_magnitude = imbalance_mass * eccentricity * (self.omega ** 2) * severity

        # Map force to a per-sample additive vibration offset.
        # At 1 Hz sampling, the ~30 Hz rotor frequency aliases to near-DC, so a sinusoidal
        # pattern is invisible in the data.  We model the observable effect directly as a
        # baseline-offset increase.
        #
        # 1 Hz daily traces contain large start/stop transients, while imbalance is
        # injected inside steadier NORMAL segments whose local std is much smaller.
        # Use an observable downsample gain so severity=10 remains visible after
        # day-level normalization without clipping the raw vibration range.
        vib_amplitude = force_magnitude * 8  # dimension: std-units per unit force

        transition = self.smooth_transition(n_points, transition_ratio=0.1)

        # Generate pseudo-shaft-phase for radial (X,Y) and axial (Z) components
        time_steps = np.arange(n_points)
        phase = 2 * np.pi * time_steps / 10  # aliased rotor phase (1 Hz sampling)

        for idx in range(n_points):
            row_idx = start_idx + idx
            amp = vib_amplitude * transition[idx]

            df.at[row_idx, self.col_x] += amp * np.sin(phase[idx]) * stats[self.col_x]['std']
            df.at[row_idx, self.col_y] += amp * np.cos(phase[idx]) * stats[self.col_y]['std']
            # Axial (Z) is less affected by radial imbalance
            df.at[row_idx, self.col_z] += amp * 0.2 * np.sin(phase[idx]) * stats[self.col_z]['std']

            if transition[idx] > 0.5:
                df.at[row_idx, self.col_t] += vib_amplitude * 0.1

            labels[row_idx] = 1

        for col in [self.col_x, self.col_y, self.col_z]:
            noise = self.rng.normal(0, stats[col]['std'] * 0.08, size=n_points)
            df.loc[start_idx:end_idx-1, col] += noise

        self.injection_log.append({
            'type': 'rotor_imbalance',
            'start': start_idx,
            'end': end_idx,
            'severity': severity,
            'force_magnitude': force_magnitude,
            'vib_amplitude': vib_amplitude
        })

        return df, labels

class MechanicalLoosenessModel(BaseAnomalyModel):
    """
    Mechanical Looseness (Structural Looseness) Model.
    
    Models the progressive loosening of motor mounting bolts or base frame.
    As bolts loosen, dynamic stiffness decreases, leading to nonlinear 
    amplification of vibrational energy, primarily in specific directions (e.g. X and Y).
    
    Physical manifestation:
    - Step-wise or exponential increase in directional vibration amplitude.
    - Large variations/variance in vibration due to instability.
    - Minor temperature effects (negligible).
    """
    def __init__(self,
                 sensor_name: str = 'BA1_Mot1',
                 random_state: Optional[int] = None):
        super().__init__(name='Mechanical_Looseness',
                        sensor_name=sensor_name,
                        random_state=random_state)
        # Stiffness degradation parameters
        self.initial_stiffness = 1.0
        self.degradation_rate = 0.05
        
    def inject(self,
               df: pd.DataFrame,
               start_ratio: float = 0.25,
               duration_ratio: float = 0.75,
               severity: float = 1.0,
               **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        self.validate_data(df)
        df = df.copy()
        n_samples = len(df)
        labels = np.zeros(n_samples, dtype=int)
        
        start_idx = int(n_samples * start_ratio)
        duration = int(n_samples * duration_ratio)
        end_idx = min(start_idx + duration, n_samples)
        
        if end_idx <= start_idx:
            return df, labels
            
        n_points = end_idx - start_idx
        stats = self.compute_stats(df)
        
        # Determine loosened axes. Looseness is strongest in radial/horizontal
        # axes, but structural slack also couples weakly into the axial channel.
        axis_weights = {
            self.col_x: 1.00,
            self.col_y: 0.85,
            self.col_z: 0.35,
        }
        target_cols = [c for c in [self.col_x, self.col_y, self.col_z] if c in df.columns]
            
        # Stiffness drops non-linearly over time
        time_steps = np.linspace(0, 1, n_points)
        stiffness_drop = 1.0 - (1.0 - np.exp(-time_steps * 3)) * 0.7 * severity
        
        # Vibration amplifies inversely proportional to dynamic stiffness
        vibration_amplification = 1.0 / np.maximum(stiffness_drop, 0.1)
        
        # Apply smooth transition
        transition = self.smooth_transition(n_points, transition_ratio=0.1)
        # Shift back so initial factor is 1.0
        amplification_factors = 1.0 + (vibration_amplification - 1.0) * transition
        
        for col in target_cols:
            baseline = df[col].iloc[start_idx:end_idx].values
            axis_weight = axis_weights.get(col, 1.0)
            # Looseness causes amplitude scaling PLUS random rattling variance
            rattle_noise = (
                self.rng.normal(0, stats[col]['std'] * 0.5 * severity * axis_weight, n_points)
                * transition
            )
            
            axis_amplification = 1.0 + (amplification_factors - 1.0) * axis_weight
            df.loc[df.index[start_idx:end_idx], col] = baseline * axis_amplification + rattle_noise
            
        labels[start_idx:end_idx] = 1
        return df, labels
