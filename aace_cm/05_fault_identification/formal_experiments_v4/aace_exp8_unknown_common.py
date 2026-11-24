"""Shared, label-free utilities for AACE-CM Exp8 unknown-mode discovery."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


EPS = 1e-12


def scenario_uid(path: Path) -> str:
    """Return a stable anonymous id without exposing the label-bearing filename."""
    digest = hashlib.sha256(path.stem.encode("utf-8")).hexdigest()[:16]
    return f"scene_{digest}"


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    return (
        values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)
    ).astype(np.float32)


def axis_time(flat: np.ndarray, channels: int) -> np.ndarray:
    """Undo time-major flattening as (sample, channel, time)."""
    flat = np.asarray(flat)
    if flat.ndim != 2 or flat.shape[1] % channels:
        raise ValueError(
            f"Expected 2-D width divisible by {channels}, got {flat.shape}"
        )
    length = flat.shape[1] // channels
    return flat.reshape(len(flat), length, channels).transpose(0, 2, 1)


def safe_correlations(values: np.ndarray) -> np.ndarray:
    """Pairwise row correlations for (samples, channels, time)."""
    values = np.asarray(values, dtype=np.float64)
    outputs = []
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            first = values[:, left] - values[:, left].mean(axis=1, keepdims=True)
            second = (
                values[:, right]
                - values[:, right].mean(axis=1, keepdims=True)
            )
            denominator = np.sqrt(
                np.sum(first**2, axis=1) * np.sum(second**2, axis=1)
            )
            outputs.append(
                np.divide(
                    np.sum(first * second, axis=1),
                    np.maximum(denominator, EPS),
                )
            )
    if not outputs:
        return np.empty((len(values), 0), dtype=np.float32)
    return np.stack(outputs, axis=1).astype(np.float32)


def spectral_shape(values: np.ndarray) -> np.ndarray:
    """Amplitude-invariant spectral shape for (samples, channels, time)."""
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(np.fft.rfft(values, axis=2))
    magnitude[..., 0] = 0.0
    normalized = magnitude / np.maximum(
        magnitude.sum(axis=2, keepdims=True), EPS
    )
    frequencies = np.linspace(0.0, 1.0, magnitude.shape[2])
    peak = normalized.argmax(axis=2) / max(magnitude.shape[2] - 1, 1)
    centroid = np.sum(
        normalized * frequencies[None, None, :], axis=2
    )
    spread = np.sqrt(
        np.sum(
            normalized
            * (frequencies[None, None, :] - centroid[:, :, None]) ** 2,
            axis=2,
        )
    )
    entropy = -np.sum(
        normalized * np.log(np.maximum(normalized, EPS)), axis=2
    ) / max(math.log(max(magnitude.shape[2], 2)), EPS)
    band_outputs = []
    for lower, upper in ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)):
        mask = (frequencies >= lower) & (frequencies < upper)
        band_outputs.append(normalized[:, :, mask].sum(axis=2))
    return np.nan_to_num(
        np.concatenate(
            [peak, centroid, spread, entropy, *band_outputs], axis=1
        )
    ).astype(np.float32)


def local_mechanism_features(
    residual_flat: np.ndarray,
    raw_flat: np.ndarray,
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Window-level residual and raw mechanism features."""
    residual = axis_time(residual_flat, channels)
    raw = axis_time(raw_flat, channels)
    mean = residual.mean(axis=2)
    standard_deviation = residual.std(axis=2) + 1e-8
    standardized = np.clip(
        (residual - mean[:, :, None]) / standard_deviation[:, :, None],
        -30.0,
        30.0,
    )
    skew = np.mean(standardized**3, axis=2)
    kurtosis = np.mean(standardized**4, axis=2)
    time = np.linspace(-1.0, 1.0, residual.shape[2])
    slope = np.sum(
        (residual - mean[:, :, None]) * time[None, None, :], axis=2
    ) / np.maximum(np.sum(time**2) * standard_deviation, 1e-8)
    endpoint = (
        np.abs(residual[:, :, -1] - residual[:, :, 0])
        / standard_deviation
    )
    rms = np.sqrt(np.mean(residual**2, axis=2))
    rms_profile = rms / np.maximum(rms.sum(axis=1, keepdims=True), EPS)
    residual_features = np.concatenate(
        [
            skew,
            kurtosis,
            slope,
            endpoint,
            safe_correlations(residual),
            rms_profile,
            spectral_shape(residual),
        ],
        axis=1,
    )
    raw_features = np.concatenate(
        [safe_correlations(raw), spectral_shape(raw)], axis=1
    )
    return (
        np.nan_to_num(residual_features).astype(np.float32),
        np.nan_to_num(raw_features).astype(np.float32),
    )


def aggregate_distribution(values: np.ndarray) -> np.ndarray:
    """Scene distribution: mean, std, and five fixed quantiles."""
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.percentile(values, [5, 25, 50, 75, 95], axis=0)
    return np.nan_to_num(
        np.concatenate([values.mean(axis=0), values.std(axis=0), *quantiles])
    ).astype(np.float32)


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return 0.0
    first = values[:-lag] - np.mean(values[:-lag])
    second = values[lag:] - np.mean(values[lag:])
    denominator = np.sqrt(np.sum(first**2) * np.sum(second**2))
    return float(np.sum(first * second) / max(float(denominator), EPS))


def series_shape(values: np.ndarray, nonnegative: bool = False) -> np.ndarray:
    """Amplitude-robust trajectory shape."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    if nonnegative:
        values = np.log1p(np.maximum(values, 0.0))
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    scale = max(float(q75 - q25), float(np.std(values)), 1e-8)
    standardized = np.clip((values - median) / scale, -30.0, 30.0)
    quantiles = np.percentile(standardized, [5, 10, 25, 50, 75, 90, 95])
    chunks = np.array_split(standardized, 10)
    chunk_means = np.asarray(
        [float(chunk.mean()) if len(chunk) else 0.0 for chunk in chunks]
    )
    lags = sorted(
        {
            max(1, int(round(len(standardized) * fraction)))
            for fraction in (0.01, 0.03, 0.07, 0.15, 0.30)
        }
    )
    acf = np.asarray([autocorrelation(standardized, lag) for lag in lags])
    acf = np.pad(acf[:5], (0, max(0, 5 - len(acf))))[:5]
    centered = standardized - standardized.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    spectrum = spectrum / max(float(spectrum.sum()), EPS)
    frequencies = np.linspace(0.0, 1.0, len(spectrum))
    peak_frequency = float(frequencies[int(np.argmax(spectrum))])
    centroid = float(np.sum(frequencies * spectrum))
    entropy = float(
        -np.sum(spectrum * np.log(np.maximum(spectrum, EPS)))
        / max(math.log(max(len(spectrum), 2)), EPS)
    )
    bands = np.asarray(
        [
            float(spectrum[(frequencies >= lo) & (frequencies < hi)].sum())
            for lo, hi in (
                (0.0, 0.05),
                (0.05, 0.15),
                (0.15, 0.35),
                (0.35, 1.01),
            )
        ]
    )
    z = centered / max(float(centered.std()), 1e-8)
    time = np.linspace(-1.0, 1.0, len(z))
    linear = float(np.corrcoef(time, z)[0, 1]) if len(z) > 2 else 0.0
    quadratic = float(np.polyfit(time, z, 2)[0]) if len(z) > 3 else 0.0
    roughness = float(np.mean(np.abs(np.diff(z)))) if len(z) > 1 else 0.0
    return np.nan_to_num(
        np.concatenate(
            [
                quantiles,
                chunk_means,
                acf,
                np.asarray(
                    [
                        peak_frequency,
                        centroid,
                        entropy,
                        *bands,
                        float(np.mean(z**3)),
                        float(np.mean(z**4)),
                        linear,
                        quadratic,
                        roughness,
                    ]
                ),
            ]
        )
    ).astype(np.float32)


def trajectory_features(
    residual_flat: np.ndarray,
    scores: np.ndarray,
    channels: int,
) -> np.ndarray:
    residual = axis_time(residual_flat, channels)
    center = residual[:, :, residual.shape[2] // 2]
    outputs = [series_shape(center[:, channel]) for channel in range(channels)]
    outputs.append(series_shape(scores, nonnegative=True))
    corr = np.corrcoef(center, rowvar=False)
    upper = corr[np.triu_indices(channels, 1)]
    rms = np.sqrt(np.mean(center**2, axis=0))
    rms_profile = rms / max(float(rms.sum()), EPS)
    return np.nan_to_num(
        np.concatenate([*outputs, upper, rms_profile])
    ).astype(np.float32)


def _best_lag(first: np.ndarray, second: np.ndarray, maximum: int = 20) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    maximum = min(maximum, max(1, len(first) // 4))
    best_correlation = 0.0
    best_lag_value = 0
    for lag in range(-maximum, maximum + 1):
        if lag < 0:
            left, right = first[-lag:], second[:lag]
        elif lag > 0:
            left, right = first[:-lag], second[lag:]
        else:
            left, right = first, second
        if len(left) < 4 or np.std(left) < EPS or np.std(right) < EPS:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if abs(correlation) > abs(best_correlation):
            best_correlation = correlation
            best_lag_value = lag
    return best_lag_value / max(maximum, 1), best_correlation


def temperature_dynamics(
    residual_flat: np.ndarray,
    channels: int,
    temperature_index: int,
) -> np.ndarray:
    """Temperature inertia and vibration-temperature lag features."""
    residual = axis_time(residual_flat, channels)
    center = residual[:, :, residual.shape[2] // 2]
    temperature = center[:, temperature_index]
    vibration_indices = [
        index for index in range(channels) if index != temperature_index
    ]
    vibration_energy = np.sqrt(
        np.mean(center[:, vibration_indices] ** 2, axis=1)
    )
    difference = np.diff(temperature)
    positive = difference[difference > 0]
    negative = difference[difference < 0]
    peak_index = int(np.argmax(np.abs(temperature)))
    lag, correlation = _best_lag(vibration_energy, temperature)
    extra = np.asarray(
        [
            float(positive.mean()) if len(positive) else 0.0,
            float(negative.mean()) if len(negative) else 0.0,
            float(np.mean(np.abs(difference))) if len(difference) else 0.0,
            peak_index / max(len(temperature) - 1, 1),
            autocorrelation(temperature, max(1, len(temperature) // 20)),
            lag,
            correlation,
        ]
    )
    return np.nan_to_num(
        np.concatenate([series_shape(temperature), extra])
    ).astype(np.float32)


def coupling_features(
    residual_flat: np.ndarray,
    channels: int,
    temperature_index: int,
) -> np.ndarray:
    residual = axis_time(residual_flat, channels)
    center = residual[:, :, residual.shape[2] // 2]
    corr = np.corrcoef(center, rowvar=False)
    upper = corr[np.triu_indices(channels, 1)]
    rms = np.sqrt(np.mean(center**2, axis=0))
    profile = rms / max(float(rms.sum()), EPS)
    vibration_indices = [
        index for index in range(channels) if index != temperature_index
    ]
    vibration_energy = np.sqrt(
        np.mean(center[:, vibration_indices] ** 2, axis=1)
    )
    lag, lag_correlation = _best_lag(
        vibration_energy, center[:, temperature_index]
    )
    return np.nan_to_num(
        np.concatenate([upper, profile, np.asarray([lag, lag_correlation])])
    ).astype(np.float32)


def severity_features(
    residual_flat: np.ndarray,
    scores: np.ndarray,
    selected: np.ndarray,
    channels: int,
    threshold: float,
) -> np.ndarray:
    residual = axis_time(residual_flat, channels)
    rms = np.sqrt(np.mean(residual**2, axis=(0, 2)))
    return np.nan_to_num(
        np.asarray(
            [
                float(scores.mean()),
                float(scores.std()),
                *np.percentile(scores, [50, 75, 90, 95, 99]).tolist(),
                float(scores.max()),
                float(np.mean(scores > threshold)),
                *rms.tolist(),
                float(len(selected) / max(len(scores), 1)),
            ],
            dtype=np.float32,
        )
    )


def event_count(centers: np.ndarray, selected: np.ndarray) -> int:
    if not len(selected):
        return 0
    selected_centers = np.sort(centers[selected])
    typical_step = max(1, int(np.median(np.diff(centers)))) if len(centers) > 1 else 1
    return int(1 + np.sum(np.diff(selected_centers) > 2 * typical_step))


def select_abnormal_windows(
    scores: np.ndarray,
    threshold: float,
    minimum_windows: int,
    maximum_fraction: float,
) -> np.ndarray:
    """Select true threshold exceedances without forcing weak scenes into Exp8."""
    scores = np.asarray(scores)
    selected = np.flatnonzero(scores > threshold)
    if len(selected) < minimum_windows:
        return np.empty(0, dtype=int)
    maximum = max(minimum_windows, int(math.ceil(maximum_fraction * len(scores))))
    if len(selected) > maximum:
        selected = np.argsort(scores)[-maximum:]
    return np.sort(selected)


def self_test() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(32, 7, 4)).astype(np.float32)
    flat = values.reshape(32, -1)
    restored = axis_time(flat, 4)
    assert np.allclose(restored.transpose(0, 2, 1), values)
    residual, raw = local_mechanism_features(flat, flat * 2, 4)
    assert residual.shape[0] == raw.shape[0] == 32
    assert np.isfinite(residual).all() and np.isfinite(raw).all()
    scores = np.linspace(0.0, 2.0, len(flat))
    selected = select_abnormal_windows(scores, 1.0, 8, 0.30)
    assert len(selected) >= 8
    assert np.isfinite(trajectory_features(flat, scores, 4)).all()
    assert np.isfinite(temperature_dynamics(flat, 4, 3)).all()
    assert np.isfinite(coupling_features(flat, 4, 3)).all()
