"""
Exp2 temperature-channel fault injection.

This script mirrors the existing Exp2 vibration-injection protocol, but injects
fault signatures into the temperature channel (BA1_T / BA2_T). It keeps the same
scenario grid and naming convention:

    {date}_{fault_type}_pct{percent}_sev{level}_{data,labels,params}

The output is intentionally written to aace_cm/04-temp_fault_data so the
existing vibration-only fault data is not overwritten. When a vibration base
directory is supplied, this script overlays the temperature response onto the
already injected X/Y/Z scenario and reuses the same fault window, producing a
single aligned X/Y/Z/T scenario.

Physical mechanisms implemented here are compact first-order thermal models:

    d(delta_T)/dt = q_fault(t) - delta_T/tau

where q_fault(t) is defined per fault type:

    thermal: Arrhenius/Q10 accelerated heat source
    imbalance: F = m e omega^2, bearing loss proportional to F^2
    mechanical_looseness: Archard wear / friction power growth
    crack: Paris-law-inspired defect growth with intermittent friction pulses

The models are deliberately conservative on the temperature channel, but the
mainline Exp2 output is four-axis data: X/Y/Z vibration perturbations from the
vibration injector plus the temperature perturbation generated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
VW_DIR = ROOT_DIR / "aace_cm"
DATA_DIR = ROOT_DIR / "DATA" / "Data preprocessing" / "AACE-CM" / "Per_second"
DEFAULT_OUT_DIR = VW_DIR / "04-temp_fault_data"
VENDOR_DIR = THIS_DIR / "_vendor_py312"

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(VW_DIR))
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from config import COMMON, PATHS, get_exp0_recommended_eval_dates, get_frozen_eval_dates  # noqa: E402
from aace_config import FaultInjectionConfig, SemanticConfig, SensorConfig  # noqa: E402

DEFAULT_DEVICES = (COMMON.device,)


def _patch_pyarrow_for_pandas() -> None:
    try:
        import pyarrow as pa
        import pyarrow.lib as palib
    except Exception:
        return
    original = getattr(pa, "unregister_extension_type", None)
    if original is None or getattr(original, "_exp2_safe", False):
        return

    def safe_unregister(name):
        try:
            return original(name)
        except palib.ArrowKeyError as exc:
            if "No type extension" in str(exc):
                return None
            raise

    safe_unregister._exp2_safe = True
    pa.unregister_extension_type = safe_unregister


EXP0_RECOMMENDED_DATES = (
    THIS_DIR / "outputs" / "exp0_data_audit" / "recommended_dates.csv"
)


@dataclass(frozen=True)
class InjectionPlan:
    date: str
    fault_type: str
    pct: float
    severity_level: int
    physics_severity: float
    rng_seed: int


SEVERITY_CAPS = dict(FaultInjectionConfig.SEVERITY_CAPS)
SEVERITY_SCALING = dict(FaultInjectionConfig.SEVERITY_SCALING)
FAULT_TYPES = list(FaultInjectionConfig.FAULT_TYPES)
GRID_PCTS = list(FaultInjectionConfig.GRID_PERCENTAGES)
GRID_SEVS = list(FaultInjectionConfig.GRID_SEVERITY_LEVELS)
GRID_MAX_SEVERITY = int(FaultInjectionConfig.GRID_MAX_SEVERITY)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _physics_severity(fault_type: str, severity_level: int) -> float:
    cap = float(SEVERITY_CAPS.get(fault_type, 1.0))
    shape = SEVERITY_SCALING.get(fault_type, "linear")
    norm = float(severity_level) / float(GRID_MAX_SEVERITY)
    if shape == "sqrt":
        return cap * np.sqrt(norm)
    if shape == "log":
        return cap * (0.01 + np.log1p(norm * 9.0) / np.log1p(9.0))
    return cap * norm


def _eligible_days(min_rows: int) -> Tuple[List[str], List[Tuple[str, int]]]:
    eligible: List[str] = []
    skipped: List[Tuple[str, int]] = []
    for path in sorted(DATA_DIR.glob("*_preprocessed.csv")):
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            n_rows = sum(1 for _ in fh) - 1
        if n_rows >= min_rows:
            eligible.append(path.name.replace("_preprocessed.csv", ""))
        else:
            skipped.append((path.name, n_rows))
    return eligible, skipped


def _read_state_labels(path: Path, n: int) -> np.ndarray:
    labels = pd.read_csv(path)
    state_col = "label" if "label" in labels.columns else "state"
    return labels[state_col].to_numpy(dtype=int)[:n]


RUNNING_ENERGY_TH = 0.52
LOAD_CHANGE_CONTEXT = 120
MAX_LOAD_CHANGE_BRIDGE = 300


def _segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    segments: List[Tuple[int, int]] = []
    start = prev = int(idx[0])
    for raw_idx in idx[1:]:
        cur = int(raw_idx)
        if cur != prev + 1:
            segments.append((start, prev + 1))
            start = cur
        prev = cur
    segments.append((start, prev + 1))
    return segments


def _derive_running_load_change_mask(
    df: pd.DataFrame,
    state_labels: np.ndarray,
    device: str,
) -> np.ndarray:
    """Recover high-energy running load changes that old labels may mark as STOPPED."""
    vib_cols = [f"{device}_{axis}" for axis in ("X", "Y", "Z") if f"{device}_{axis}" in df.columns]
    if len(vib_cols) < 3:
        return np.zeros(len(state_labels), dtype=bool)
    x = df[vib_cols].iloc[:len(state_labels)].to_numpy(dtype=float)
    energy = np.sum(np.abs(x), axis=1)
    stopped = np.isin(state_labels, (SemanticConfig.STOPPED, SemanticConfig.SHUTDOWN))
    high_energy = energy >= RUNNING_ENERGY_TH
    out = np.zeros(len(state_labels), dtype=bool)
    running_states = (SemanticConfig.COLD_START, SemanticConfig.NORMAL,
                      SemanticConfig.HOT_START, SemanticConfig.LOAD_CHANGE)

    for start, end in _segments_from_mask(stopped):
        candidate = np.zeros(len(state_labels), dtype=bool)
        candidate[start:end] = high_energy[start:end]
        if not candidate[start:end].any():
            continue
        pre_s = max(0, start - LOAD_CHANGE_CONTEXT)
        post_e = min(len(state_labels), end + LOAD_CHANGE_CONTEXT)
        before_y = state_labels[pre_s:start]
        after_y = state_labels[end:post_e]
        before_e = energy[pre_s:start]
        after_e = energy[end:post_e]
        before_running = (
            np.isin(before_y, running_states).any()
            or (len(before_e) > 0 and np.mean(before_e >= RUNNING_ENERGY_TH) >= 0.35)
        )
        after_running = (
            np.isin(after_y, running_states).any()
            or (len(after_e) > 0 and np.mean(after_e >= RUNNING_ENERGY_TH) >= 0.35)
        )
        if before_running and after_running:
            out |= candidate
    return out


def _merge_segments_across_continuity(
    segments: List[Tuple[int, int]],
    continuity_mask: np.ndarray,
    max_bridge_gap: int = MAX_LOAD_CHANGE_BRIDGE,
) -> List[Tuple[int, int]]:
    if not segments:
        return []
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = segments[0]
    for next_s, next_e in segments[1:]:
        gap = next_s - cur_e
        can_bridge = (
            0 < gap <= max_bridge_gap
            and bool(np.all(continuity_mask[cur_e:next_s]))
        )
        if can_bridge:
            cur_e = next_e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = next_s, next_e
    merged.append((cur_s, cur_e))
    return merged


def _select_normal_segments(
    state_labels: np.ndarray,
    pct: float,
    rng: np.random.Generator,
    continuity_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]], int, int]:
    normal_indices = np.where(state_labels == SemanticConfig.NORMAL)[0]
    n_total = len(state_labels)
    if len(normal_indices) == 0:
        return np.zeros(n_total, dtype=bool), [], 0, 0

    n_normal = len(normal_indices)
    n_inject_total = max(1, int(n_normal * pct))

    global_start = int(n_normal * 0.10)
    global_end = int(n_normal * 0.80)
    available = global_end - global_start
    if available < n_inject_total:
        start_flat = int(rng.integers(0, max(1, n_normal - n_inject_total + 1)))
    else:
        start_flat = int(rng.integers(global_start, global_end - n_inject_total + 1))
    end_flat = start_flat + n_inject_total

    mask = np.zeros(n_total, dtype=bool)
    mask[normal_indices[start_flat:end_flat]] = True

    segments = _segments_from_mask(mask)
    if continuity_mask is not None:
        continuity_mask = np.asarray(continuity_mask, dtype=bool)[:n_total]
        segments = _merge_segments_across_continuity(segments, continuity_mask)
        mask = np.zeros(n_total, dtype=bool)
        for start, end in segments:
            mask[start:end] = True

    total_injected = int(sum(end - start for start, end in segments))
    return mask, segments, n_inject_total, total_injected


def _thermal_response(
    heat_source: np.ndarray,
    tau: float,
    dt: float = 1.0,
) -> np.ndarray:
    delta = np.zeros_like(heat_source, dtype=float)
    if len(delta) == 0:
        return delta
    for i in range(1, len(delta)):
        delta[i] = delta[i - 1] + dt * (heat_source[i] - delta[i - 1] / tau)
    return np.maximum(delta, 0.0)


def _cooling_tau_multiplier(state: int) -> float:
    """State-dependent cooling inertia.

    STOPPED/SHUTDOWN cool more slowly because oil circulation is reduced; this
    also prevents artificial hard drops when a fault window crosses semantic
    state boundaries.
    """
    if state in (SemanticConfig.STOPPED, SemanticConfig.SHUTDOWN):
        return 2.4
    if state in (SemanticConfig.COLD_START, SemanticConfig.HOT_START, SemanticConfig.LOAD_CHANGE):
        return 1.35
    return 1.0


def _thermal_inertia_response(
    target_delta: np.ndarray,
    state_labels: np.ndarray,
    tau_rise: float,
    tau_cool: float,
    dt: float = 1.0,
) -> np.ndarray:
    """First-order thermal inertia response to a target fault heat load.

    The target may go to zero immediately after the fault source disappears,
    but the observed temperature delta decays with tau_cool instead of being
    clipped to zero at the label boundary.
    """
    target = np.asarray(target_delta, dtype=float)
    states = np.asarray(state_labels, dtype=int)[:len(target)]
    response = np.zeros_like(target, dtype=float)
    if len(response) == 0:
        return response

    for i in range(1, len(response)):
        prev = response[i - 1]
        tgt = max(float(target[i]), 0.0)
        if tgt >= prev:
            tau = max(float(tau_rise), 1.0)
        else:
            tau = max(float(tau_cool) * _cooling_tau_multiplier(int(states[i])), 1.0)
        alpha = 1.0 - np.exp(-float(dt) / tau)
        response[i] = prev + alpha * (tgt - prev)
    return np.maximum(response, 0.0)


def _scale_to_peak(delta: np.ndarray, target_peak: float) -> np.ndarray:
    peak = float(np.nanmax(delta)) if len(delta) else 0.0
    if peak <= 1e-12:
        return np.zeros_like(delta)
    return delta * (target_peak / peak)


def _duration_limited_peak(steady_peak: float, n_steps: int, tau: float) -> float:
    # A fault segment shorter than the thermal time constant should not reach
    # the same peak as a long degradation episode.
    duration_factor = 1.0 - np.exp(-float(n_steps) / max(float(tau), 1.0))
    return float(steady_peak) * float(np.clip(duration_factor, 0.05, 1.0))


def _fault_temperature_delta(
    baseline_temp: np.ndarray,
    fault_type: str,
    physics_severity: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, dict]:
    n = len(baseline_temp)
    if n == 0:
        return np.zeros(0, dtype=float), {}

    x = np.linspace(0.0, 1.0, n)
    temp = baseline_temp.astype(float)
    temp_ref = float(np.nanmedian(temp))
    temp_k = np.clip(temp + 273.15, 250.0, 380.0)
    cap = float(SEVERITY_CAPS.get(fault_type, 1.0))
    s = min(1.0, float(physics_severity) / max(cap, 1e-12))

    if fault_type == "thermal":
        # Arrhenius/Q10 approximation: warmer baseline increases the thermal
        # equilibrium, but the observed channel reaches it only through inertia.
        q10_gamma = np.log(2.0)
        arrhenius = np.exp(q10_gamma * (temp - temp_ref) / 10.0)
        growth = 1.0 - np.exp(-2.2 * x)
        steady_peak = (2.5 + 14.0 * s) * float(np.nanmedian(arrhenius))
        target_peak = steady_peak
        delta = target_peak * growth
        tau_rise = 780.0 + 720.0 * (1.0 - s)
        tau_cool = tau_rise * 1.65
        model = "Arrhenius/Q10 heat accumulation"
    elif fault_type == "imbalance":
        # F = m e omega^2. The extra bearing heat is modeled as proportional
        # to F^2, with slow modulation after downsampling.
        force_ratio = max(s, 1e-6)
        modulation = 1.0 + 0.05 * np.sin(2.0 * np.pi * (1.0 * x + rng.random()))
        growth = 1.0 - np.exp(-2.5 * x)
        steady_peak = 0.8 + 5.2 * force_ratio**2
        target_peak = steady_peak
        delta = target_peak * growth * modulation
        tau_rise = 620.0
        tau_cool = 1050.0
        model = "unbalance force squared loss"
    elif fault_type == "mechanical_looseness":
        # Archard wear V = k W s / H. Friction power increases as looseness
        # grows; contact events are broad thermal loads, not instantaneous
        # temperature jumps.
        wear_growth = np.sqrt(np.maximum(x, 1e-6))
        steady_peak = 1.2 + 9.0 * s
        delta = steady_peak * (0.25 + 0.75 * wear_growth)
        if n >= 20:
            burst_count = max(1, int(2 + 5 * s))
            centers = rng.integers(0, n, size=burst_count)
            width = max(60, int(0.08 * n))
            idx = np.arange(n)
            for center in centers:
                delta += (0.3 + 1.4 * s) * np.exp(-0.5 * ((idx - center) / width) ** 2)
        target_peak = float(np.nanmax(delta))
        tau_rise = 650.0
        tau_cool = 1150.0
        model = "Archard wear and frictional heating"
    elif fault_type == "crack":
        # Paris law da/dN = C (Delta K)^m implies accelerating defect growth.
        # Temperature rise is modest early and accelerates near the segment end.
        paris_growth = np.maximum(x, 1e-6) ** (2.2 + 0.8 * s)
        steady_peak = 0.7 + 6.2 * s
        delta = steady_peak * (0.15 + 0.85 * paris_growth)
        if n >= 30:
            pulse_count = max(1, int(1 + 4 * s))
            centers = rng.integers(int(0.2 * n), n, size=pulse_count)
            width = max(45, int(0.06 * n))
            idx = np.arange(n)
            for center in centers:
                delta += (0.2 + 1.0 * s) * np.exp(-0.5 * ((idx - center) / width) ** 2)
        target_peak = float(np.nanmax(delta))
        tau_rise = 580.0
        tau_cool = 980.0
        model = "Paris-law defect growth with friction pulses"
    else:
        raise ValueError(f"Unsupported fault type: {fault_type}")

    delta = _scale_to_peak(np.maximum(delta, 0.0), target_peak=target_peak)

    metadata = {
        "mechanism": model,
        "thermal_equation": "d(delta_T)/dt = (delta_T_target(t) - delta_T)/tau_state",
        "tau_rise_seconds": float(tau_rise),
        "tau_cool_seconds": float(tau_cool),
        "target_peak_delta_c": float(target_peak),
        "target_mean_delta_c": float(np.nanmean(delta)),
    }
    return delta, metadata


def _inject_temperature(
    df: pd.DataFrame,
    state_labels: np.ndarray,
    device: str,
    plan: InjectionPlan,
    preset_fault_labels: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    temp_col = f"{device}_T"
    if temp_col not in df.columns:
        raise KeyError(f"Missing temperature column: {temp_col}")

    rng = np.random.default_rng(plan.rng_seed)
    if preset_fault_labels is not None:
        preset = np.asarray(preset_fault_labels, dtype=int)[:len(df)]
        fault_mask = preset == 1
        segments = _segments_from_mask(fault_mask)
        n_target = int(fault_mask.sum())
        n_actual = int(fault_mask.sum())
        derived_load_change = _derive_running_load_change_mask(df, state_labels, device)
        selection_source = "vibration_fault_labels"
    else:
        derived_load_change = _derive_running_load_change_mask(df, state_labels, device)
        continuity_mask = (
            (state_labels == SemanticConfig.NORMAL)
            | (state_labels == SemanticConfig.LOAD_CHANGE)
            | derived_load_change
        )
        fault_mask, segments, n_target, n_actual = _select_normal_segments(
            state_labels=state_labels,
            pct=plan.pct,
            rng=rng,
            continuity_mask=continuity_mask,
        )
        selection_source = "temperature_normal_segment_sampler"

    injected = df.copy()
    labels = np.zeros(len(df), dtype=int)
    segment_meta = []
    target_delta = np.zeros(len(df), dtype=float)
    tau_rise_values: List[float] = []
    tau_cool_values: List[float] = []

    for start, end in segments:
        baseline = df[temp_col].iloc[start:end].to_numpy(dtype=float)
        segment_target, meta = _fault_temperature_delta(
            baseline_temp=baseline,
            fault_type=plan.fault_type,
            physics_severity=plan.physics_severity,
            rng=rng,
        )
        target_delta[start:end] = np.maximum(target_delta[start:end], segment_target)
        tau_rise_values.append(float(meta.get("tau_rise_seconds", 900.0)))
        tau_cool_values.append(float(meta.get("tau_cool_seconds", 1500.0)))
        labels[start:end] = 1
        segment_meta.append({
            "start": int(start),
            "end": int(end),
            "length": int(end - start),
            **meta,
        })

    tau_rise = float(np.nanmedian(tau_rise_values)) if tau_rise_values else 900.0
    tau_cool = float(np.nanmedian(tau_cool_values)) if tau_cool_values else 1500.0
    total_delta = _thermal_inertia_response(
        target_delta=target_delta,
        state_labels=state_labels,
        tau_rise=tau_rise,
        tau_cool=tau_cool,
    )
    injected[temp_col] = df[temp_col].to_numpy(dtype=float) + total_delta

    for meta in segment_meta:
        s0 = int(meta["start"])
        e0 = int(meta["end"])
        seg_delta = total_delta[s0:e0]
        meta["actual_peak_delta_c"] = float(np.nanmax(seg_delta)) if len(seg_delta) else 0.0
        meta["actual_mean_delta_c"] = float(np.nanmean(seg_delta)) if len(seg_delta) else 0.0

    info = {
        "n_normal": int((state_labels == SemanticConfig.NORMAL).sum()),
        "n_inject_target": int(n_target),
        "total_injected": int(n_actual),
        "n_derived_load_change": int(derived_load_change.sum()),
        "continuity_states": ["NORMAL", "LOAD_CHANGE", "derived_running_load_change"],
        "fault_window_source": selection_source,
        "max_load_change_bridge": int(MAX_LOAD_CHANGE_BRIDGE),
        "injected_segments": [[int(s), int(e)] for s, e in segments],
        "segment_meta": segment_meta,
        "temperature_column": temp_col,
        "thermal_response_scope": "full_day_inertial_response",
        "thermal_equation": "d(delta_T)/dt = (delta_T_target(t) - delta_T)/tau_state",
        "tau_rise_seconds": float(tau_rise),
        "tau_cool_seconds": float(tau_cool),
        "cooling_tau_state_multipliers": {
            "NORMAL": 1.0,
            "COLD_START": 1.35,
            "HOT_START": 1.35,
            "LOAD_CHANGE": 1.35,
            "STOPPED": 2.4,
            "SHUTDOWN": 2.4,
        },
        "delta_peak_c": float(np.nanmax(total_delta)) if len(total_delta) else 0.0,
        "delta_mean_fault_c": float(np.nanmean(total_delta[fault_mask])) if fault_mask.any() else 0.0,
        "delta_mean_tail_c": float(np.nanmean(total_delta[~fault_mask])) if (~fault_mask).any() else 0.0,
        "delta_peak_tail_c": float(np.nanmax(total_delta[~fault_mask])) if (~fault_mask).any() else 0.0,
    }
    return injected, labels, info


def _lean_output(df: pd.DataFrame, device: str) -> pd.DataFrame:
    cols = ["Timestamp"]
    if "time_idx" in df.columns:
        cols.append("time_idx")
    for suffix in ("T", "X", "Y", "Z"):
        col = f"{device}_{suffix}"
        if col in df.columns:
            cols.append(col)
    return df[cols].copy()


def _write_scenario_from_frame(
    out_dir: Path,
    df: pd.DataFrame,
    state_labels: np.ndarray,
    source_file: Path,
    labels_path: Path,
    device: str,
    plan: InjectionPlan,
    overwrite: bool,
    preset_fault_labels: np.ndarray | None = None,
    base_scenario_file: Path | None = None,
    base_labels_file: Path | None = None,
) -> dict:
    file_stem = source_file.name.replace("_preprocessed.csv", "")
    scenario = f"{file_stem}_{plan.fault_type}_pct{int(plan.pct * 100)}_sev{plan.severity_level}"
    data_out = out_dir / device / f"{scenario}_data.parquet"
    labels_out = out_dir / device / f"{scenario}_labels.parquet"
    params_out = out_dir / device / f"{scenario}_params.json"

    if data_out.exists() and labels_out.exists() and params_out.exists() and not overwrite:
        return {"scenario": scenario, "status": "skipped_existing"}

    injected, fault_labels, info = _inject_temperature(
        df, state_labels, device, plan,
        preset_fault_labels=preset_fault_labels,
    )
    lean = _lean_output(injected, device)

    data_out.parent.mkdir(parents=True, exist_ok=True)
    _patch_pyarrow_for_pandas()
    lean.to_parquet(data_out, index=False)
    pd.DataFrame({
        "fault_label": fault_labels,
        "state": state_labels,
    }).to_parquet(labels_out, index=False)

    params = {
        "bearing": device,
        "fault_type": plan.fault_type,
        "channel": f"{device}_T",
        "percentage": float(plan.pct),
        "severity_level": int(plan.severity_level),
        "physics_severity": float(plan.physics_severity),
        "n_fault_samples": int(fault_labels.sum()),
        "source_file": source_file.name,
        "labels_file": labels_path.name,
        "rng_seed": int(plan.rng_seed),
        "output_kind": "vibration_temperature_fault_data" if base_scenario_file else "temperature_fault_data",
        "base_scenario_file": str(base_scenario_file) if base_scenario_file else None,
        "base_labels_file": str(base_labels_file) if base_labels_file else None,
        **info,
    }
    params_out.write_text(json.dumps(params, indent=2), encoding="utf-8")

    return {
        "scenario": scenario,
        "status": "written",
        "n_fault_samples": int(fault_labels.sum()),
        "delta_peak_c": params["delta_peak_c"],
        "delta_mean_fault_c": params["delta_mean_fault_c"],
    }


def _write_scenario(
    out_dir: Path,
    source_file: Path,
    labels_path: Path,
    device: str,
    plan: InjectionPlan,
    overwrite: bool,
) -> dict:
    df = pd.read_csv(source_file)
    state_labels = _read_state_labels(labels_path, len(df))
    n = min(len(df), len(state_labels))
    df = df.iloc[:n].reset_index(drop=True)
    state_labels = state_labels[:n]
    return _write_scenario_from_frame(
        out_dir=out_dir,
        df=df,
        state_labels=state_labels,
        source_file=source_file,
        labels_path=labels_path,
        device=device,
        plan=plan,
        overwrite=overwrite,
    )


def _scenario_name(date: str, fault_type: str, pct: float, severity: int) -> str:
    return f"{date}_{fault_type}_pct{int(pct * 100)}_sev{severity}"


def _load_vibration_base_frame(
    base_dir: Path,
    device: str,
    scenario: str,
    source_day: pd.DataFrame,
    fallback_state_labels: np.ndarray,
    temp_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Path, Path] | None:
    data_path = base_dir / device / f"{scenario}_data.parquet"
    labels_path = base_dir / device / f"{scenario}_labels.parquet"
    if not data_path.exists() or not labels_path.exists():
        return None

    _patch_pyarrow_for_pandas()
    base_df = pd.read_parquet(data_path)
    base_labels = pd.read_parquet(labels_path)
    if "fault_label" not in base_labels.columns:
        raise ValueError(f"Missing fault_label in vibration base labels: {labels_path}")

    n = min(len(base_df), len(source_day), len(base_labels), len(fallback_state_labels))
    frame = base_df.iloc[:n].reset_index(drop=True).copy()
    if temp_col not in source_day.columns:
        raise KeyError(f"Missing temperature column in source day: {temp_col}")
    frame[temp_col] = source_day[temp_col].iloc[:n].to_numpy()

    if "state" in base_labels.columns:
        state_labels = base_labels["state"].to_numpy(dtype=int)[:n]
    else:
        state_labels = fallback_state_labels[:n]
    fault_labels = base_labels["fault_label"].to_numpy(dtype=int)[:n]
    return frame, state_labels, fault_labels, data_path, labels_path


def _parse_csv_floats(value: str | None, default: Sequence[float]) -> List[float]:
    if not value:
        return list(default)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_ints(value: str | None, default: Sequence[int]) -> List[int]:
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strings(value: str | None, default: Sequence[str]) -> List[str]:
    if not value:
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def _load_exp0_recommended_dates(path: Path = EXP0_RECOMMENDED_DATES) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Exp0 recommended date file not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Exp0 recommended date file must contain a 'date' column: {path}")
    return df["date"].astype(str).tolist()


def _exp0_recommended_eval_dates() -> List[str]:
    dates = get_exp0_recommended_eval_dates()
    if dates:
        return dates
    all_dates = _load_exp0_recommended_dates()
    return all_dates[len(all_dates) // 2:]


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    vibration_base_dir = Path(args.vibration_base_dir).resolve() if getattr(args, "vibration_base_dir", "") else None
    devices = _parse_csv_strings(args.devices, DEFAULT_DEVICES)
    fault_types = _parse_csv_strings(args.fault_types, FAULT_TYPES)
    pcts = _parse_csv_floats(args.pcts, GRID_PCTS)
    severities = _parse_csv_ints(args.severities, GRID_SEVS)

    eligible, skipped = _eligible_days(min_rows=args.min_rows)
    if args.dates:
        requested = set(_parse_csv_strings(args.dates, []))
        dates = [d for d in eligible if d in requested]
    elif args.date_mode == "exp0_recommended_eval":
        exp0_dates = set(_exp0_recommended_eval_dates())
        dates = [d for d in eligible if d in exp0_dates]
    elif args.date_mode == "exp0_recommended":
        exp0_dates = set(_load_exp0_recommended_dates())
        dates = [d for d in eligible if d in exp0_dates]
    elif args.date_mode == "frozen_eval":
        frozen_eval = set(get_frozen_eval_dates())
        dates = [d for d in eligible if d in frozen_eval]
    elif args.date_mode == "frozen_all":
        from config import get_frozen_all_dates

        frozen_all = set(get_frozen_all_dates())
        dates = [d for d in eligible if d in frozen_all]
    elif args.date_mode == "all":
        dates = list(eligible)
    elif args.date_mode == "latest":
        n = args.num_days if args.num_days > 0 else len(eligible)
        dates = list(eligible[-n:])
    elif args.date_mode == "first":
        n = args.num_days if args.num_days > 0 else len(eligible)
        dates = list(eligible[:n])
    else:
        raise ValueError(f"Unknown date mode: {args.date_mode}")

    if args.dates:
        missing = sorted(requested - set(dates))
        if missing:
            print(f"  requested dates not eligible: {missing}")
    if args.limit_days:
        dates = dates[:args.limit_days]

    if getattr(args, "clean_output", False) and not args.dry:
        removed = 0
        for device in devices:
            device_dir = out_dir / device
            if not device_dir.exists():
                continue
            for pattern in ("*_data.parquet", "*_labels.parquet", "*_params.json"):
                for path in device_dir.glob(pattern):
                    path.unlink()
                    removed += 1
        print(f"  cleaned old temperature scenario files: {removed}")

    print("=" * 72)
    print("Exp2 temperature fault injection")
    print(f"  source: {DATA_DIR}")
    print(f"  labels: {PATHS.vw_labels_dir}")
    print(f"  output: {out_dir}")
    print(f"  vibration base: {vibration_base_dir if vibration_base_dir else '<none>'}")
    print(f"  devices: {devices}")
    print(f"  date_mode: {args.date_mode}")
    print(f"  dates: {len(dates)}")
    if dates:
        print(f"  date range: {dates[0]} .. {dates[-1]}")
    print(f"  grid: {len(fault_types)} fault types x {len(pcts)} pct x {len(severities)} severity")
    if skipped:
        print(f"  skipped by row count: {len(skipped)}")
    print("=" * 72)

    if args.dry:
        for device in devices:
            for date in dates:
                print(f"[plan] {device} {date}: {len(fault_types) * len(pcts) * len(severities)} scenarios")
        return

    rows = []
    for device in devices:
        temp_col = f"{device}_T"
        for date in dates:
            source_file = DATA_DIR / f"{date}_preprocessed.csv"
            labels_path = PATHS.vw_labels_dir / device / f"{date}_labels.csv"
            if not source_file.exists() or not labels_path.exists():
                print(f"[skip] {device} {date}: missing source or labels")
                continue
            header = pd.read_csv(source_file, nrows=0)
            if temp_col not in header.columns:
                print(f"[skip] {device} {date}: missing {temp_col}")
                continue
            df_day = pd.read_csv(source_file)
            state_labels = _read_state_labels(labels_path, len(df_day))
            n = min(len(df_day), len(state_labels))
            df_day = df_day.iloc[:n].reset_index(drop=True)
            state_labels = state_labels[:n]

            total_for_day = len(fault_types) * len(pcts) * len(severities)
            count = 0
            for fault_type in fault_types:
                for pct in pcts:
                    for severity in severities:
                        scenario = _scenario_name(date, fault_type, pct, severity)
                        seed = _stable_seed(source_file, device, fault_type, pct, severity)
                        plan = InjectionPlan(
                            date=date,
                            fault_type=fault_type,
                            pct=pct,
                            severity_level=severity,
                            physics_severity=_physics_severity(fault_type, severity),
                            rng_seed=seed,
                        )
                        frame = df_day
                        frame_state_labels = state_labels
                        preset_fault_labels = None
                        base_data_path = None
                        base_labels_path = None
                        if vibration_base_dir is not None:
                            base = _load_vibration_base_frame(
                                base_dir=vibration_base_dir,
                                device=device,
                                scenario=scenario,
                                source_day=df_day,
                                fallback_state_labels=state_labels,
                                temp_col=temp_col,
                            )
                            if base is None:
                                raise FileNotFoundError(
                                    f"Missing vibration base for four-axis scenario: {scenario} "
                                    f"under {vibration_base_dir / device}"
                                )
                            frame, frame_state_labels, preset_fault_labels, base_data_path, base_labels_path = base
                        result = _write_scenario_from_frame(
                            out_dir=out_dir,
                            df=frame,
                            state_labels=frame_state_labels,
                            source_file=source_file,
                            labels_path=labels_path,
                            device=device,
                            plan=plan,
                            overwrite=args.overwrite,
                            preset_fault_labels=preset_fault_labels,
                            base_scenario_file=base_data_path,
                            base_labels_file=base_labels_path,
                        )
                        result.update({
                            "device": device,
                            "date": date,
                            "fault_type": fault_type,
                            "pct": pct,
                            "severity": severity,
                            "physics_severity": plan.physics_severity,
                        })
                        rows.append(result)
                        count += 1
                        if count % 25 == 0 or count == total_for_day:
                            print(f"[{device} {date}] {count}/{total_for_day}")

    if rows:
        summary = pd.DataFrame(rows)
        summary_path = out_dir / "temperature_fault_injection_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        written = int((summary["status"] == "written").sum())
        skipped_existing = int((summary["status"] == "skipped_existing").sum())
        print(f"\nDone. written={written}, skipped_existing={skipped_existing}")
        print(f"Summary: {summary_path}")
    else:
        print("\nNo scenarios were generated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp2 temperature-channel fault injection")
    parser.add_argument("--dry", action="store_true", help="Show generation plan only")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scenario files")
    parser.add_argument("--clean-output", action="store_true", help="Delete existing scenario files for selected devices before generation")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument(
        "--vibration-base-dir",
        default="",
        help="Optional vibration fault-data directory. When set, temperature is overlaid on matching XYZ scenarios.",
    )
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES), help="Comma-separated devices")
    parser.add_argument("--fault-types", default=",".join(FAULT_TYPES), help="Comma-separated fault types")
    parser.add_argument("--pcts", default=None, help="Comma-separated injection percentages")
    parser.add_argument("--severities", default=None, help="Comma-separated severity levels")
    parser.add_argument("--dates", default=None, help="Comma-separated dates to include")
    parser.add_argument(
        "--date-mode",
        default="exp0_recommended_eval",
        choices=["exp0_recommended_eval", "exp0_recommended", "frozen_eval", "frozen_all", "all", "latest", "first"],
        help="Date selection strategy when --dates is not provided",
    )
    parser.add_argument("--num-days", type=int, default=0, help="Number of days for latest/first date modes")
    parser.add_argument("--limit-days", type=int, default=0, help="Optional cap on number of dates")
    parser.add_argument("--min-rows", type=int, default=SensorConfig.MIN_ROWS, help="Minimum rows per source day")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
