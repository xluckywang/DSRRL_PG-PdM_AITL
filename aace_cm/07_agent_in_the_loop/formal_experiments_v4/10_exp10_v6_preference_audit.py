#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp10-v6: guarded preference selection and executable-action auditing.

This revision deliberately separates three protocols that were conflated in
earlier Exp10 runs:

1. ``reproduce`` is a bridge to the published Exp9-v6 result.  It forces the
   original RMS fallback used by the 2026-07-01 server run and retains the
   legacy test-checkpoint rule only for reproducibility diagnosis.
2. ``validation/select/final`` is the leak-free Exp10-A protocol.  It retains
   the same RMS risk representation as formal Exp9, selects one common episode
   budget on validation data, and evaluates the held-out test set once.  A
   numerical trust-region search proposes bounded, interpretable QR-DQN
   decision controls; the LLM may select only from a feasible shortlist.
3. ``audit`` is Exp10-B.  Only cooldown-free decision-eligible states are
   audited.  Requested and executed actions are recorded separately, all
   three seeds are retained, and horizons 8/16/32 are reported with both
   stratified and population-weighted summaries.

Exp10-C is intentionally outside this script and is the only Exp10 component
allowed to consume Exp7 DSHI as structured explanatory evidence.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = THIS_DIR / "outputs"
EXP9_PATH = THIS_DIR / "09_exp9_qrdqn_v6_model_comparison.py"
UTIL_PATH = THIS_DIR / "10_exp10_v3_formal_agent_loop.py"

RMS_ONLY_TAG = "__exp9_20260701_rms_protocol__"
DEFAULT_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
FORMAL_SEEDS = (11, 22, 33)
FORMAL_EPISODE_GRID = (100, 300, 500, 1000)
TRAIN_ENV_SEED = 100
TEST_ENV_SEED = 500
UPDATE_FREQ = 4
ACTION_ORDER = ("Continue", "Inspect", "Preventive", "Immediate")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


os.environ.setdefault("EXP7_DSHI_CACHE_TAG", RMS_ONLY_TAG)
os.environ.setdefault("EXP7_DSHI_STRICT", "0")
exp9 = _load_module(EXP9_PATH, "exp9_runtime_for_exp10v6")
util = _load_module(UTIL_PATH, "exp10_util_for_exp10v6")
torch = exp9.torch


@dataclass(frozen=True)
class DecisionConfig:
    """Interpretable deployment controls applied to QR-DQN quantiles."""

    cvar_alpha: float = 1.0
    maintenance_bias: float = 0.0
    urgency_bias: float = 0.0

    @property
    def key(self) -> str:
        return f"a{self.cvar_alpha:.3f}_m{self.maintenance_bias:+.3f}_u{self.urgency_bias:+.3f}"


@dataclass(frozen=True)
class PreferenceProfile:
    key: str
    display_name: str
    objective: str
    constraints: str


PROFILES: Dict[str, PreferenceProfile] = {
    "safety": PreferenceProfile(
        "safety",
        "Safety-first Agent",
        "Minimize unmitigated rate while preserving reward and limiting extra interventions.",
        "reward >= baseline - 0.25; Events/100 <= baseline + 0.30; improve unmitigated by >= 0.10 percentage points.",
    ),
    "balanced": PreferenceProfile(
        "balanced",
        "Balanced Agent",
        "Maximize mean reward under safety and intervention non-inferiority guards.",
        "unmitigated <= baseline + 0.10 percentage points; Events/100 <= baseline + 0.20; improve reward by >= 0.05.",
    ),
    "utilization": PreferenceProfile(
        "utilization",
        "Utilization-first Agent",
        "Minimize intervention events while preserving safety and reward.",
        "unmitigated <= baseline + 0.20 percentage points; reward >= baseline - 0.25; reduce Events/100 by >= 0.10.",
    ),
}


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _metric_view(result: Dict[str, Any]) -> Dict[str, float]:
    keys = (
        "mean_reward",
        "std_reward",
        "unmitigated_rate",
        "maintenance_rate",
        "intervention_event_rate",
        "events_per_100_windows",
        "low_risk_intervention_event_rate",
        "downtime_cost",
        "failure_response_rate",
        "warning_response_rate",
    )
    return {key: float(result.get(key, 0.0)) for key in keys}


def _aggregate_metric_views(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate an empty metric list.")
    out: Dict[str, float] = {}
    for key in rows[0]:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        out[key] = float(values.mean())
        out[f"{key}_seed_std"] = float(values.std(ddof=0))
    return out


def _set_risk_source(mode: str) -> None:
    if mode != "rms":
        raise ValueError(f"Exp10-A/B only support the formal Exp9 RMS protocol, received: {mode}")
    os.environ["EXP7_DSHI_CACHE_TAG"] = RMS_ONLY_TAG
    os.environ["EXP7_DSHI_STRICT"] = "0"


def _load_context(mode: str, fast: bool, artifact_dir: Path):
    _set_risk_source(mode)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scenarios = exp9.load_scenarios(
        fast=fast,
        device=exp9.COMMON.device,
        crop_fault_neighborhood=False,
    )
    train_full, test = exp9.split_scenarios(scenarios)
    train_full, test, diagnostic = exp9.attach_exp8_probabilities(train_full, test, artifact_dir)
    sources = {
        source: sum(s.risk_source == source for s in scenarios)
        for source in sorted({s.risk_source for s in scenarios})
    }
    if not fast:
        if len(scenarios) != 144 or len(train_full) != 100 or len(test) != 44:
            raise RuntimeError(
                f"Formal split mismatch: total/train/test={len(scenarios)}/{len(train_full)}/{len(test)}"
            )
        expected = {"rms_fallback": 144}
        if sources != expected:
            raise RuntimeError(f"Risk-source mismatch for {mode}: expected {expected}, found {sources}")
    return scenarios, train_full, test, diagnostic, sources


def _validation_split(train_full: Sequence[Any], ratio: float, seed: int, fast: bool):
    groups = sorted({(int(s.severity_level), int(s.percentage)) for s in train_full})
    if len(groups) < 2:
        if not fast:
            raise RuntimeError("Formal validation requires at least two severity/percentage groups.")
        order = np.arange(len(train_full))
        np.random.default_rng(seed).shuffle(order)
        n_val = max(1, min(len(order) - 1, int(round(len(order) * ratio))))
        val_ids = {int(i) for i in order[:n_val]}
        return (
            [s for i, s in enumerate(train_full) if i not in val_ids],
            [s for i, s in enumerate(train_full) if i in val_ids],
            ["fast-scenario-fallback"],
        )
    order = np.asarray(groups, dtype=object)
    np.random.default_rng(seed).shuffle(order)
    n_val = max(1, min(len(groups) - 1, int(round(len(groups) * ratio))))
    val_groups = {(int(x[0]), int(x[1])) for x in order[:n_val]}
    train_core = [s for s in train_full if (int(s.severity_level), int(s.percentage)) not in val_groups]
    val = [s for s in train_full if (int(s.severity_level), int(s.percentage)) in val_groups]
    labels = [f"severity={a},percentage={b}" for a, b in sorted(val_groups)]
    return train_core, val, labels


def _load_validation_context(fast: bool, artifact_dir: Path, ratio: float, split_seed: int):
    """Build validation evidence without fitting Exp8 on validation labels."""
    _set_risk_source("rms")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scenarios = exp9.load_scenarios(
        fast=fast,
        device=exp9.COMMON.device,
        crop_fault_neighborhood=False,
    )
    train_full_raw, _ = exp9.split_scenarios(scenarios)
    train_core_raw, val_raw, labels = _validation_split(train_full_raw, ratio, split_seed, fast)
    train_core, val, diagnostic = exp9.attach_exp8_probabilities(train_core_raw, val_raw, artifact_dir)
    sources = {
        source: sum(s.risk_source == source for s in scenarios)
        for source in sorted({s.risk_source for s in scenarios})
    }
    if not fast:
        if len(scenarios) != 144 or sources != {"rms_fallback": 144}:
            raise RuntimeError(f"Formal validation risk-source mismatch: total={len(scenarios)} sources={sources}")
    return train_core, val, labels, diagnostic, sources


def _make_exact_agent(raw_state_dim: int, seed: int, fast: bool):
    """Use the exact Exp9 QR-DQN constructor and formal hyperparameters."""
    return exp9.TorchQRDQNAgent(
        raw_state_dim=raw_state_dim,
        encoder=exp9._plain_dqn_encoder,
        mode="plain",
        lr=4e-4,
        gamma=0.95,
        epsilon_decay=0.9996,
        batch_size=48 if fast else 96,
        replay_size=70000,
        target_update=70,
        seed=int(seed),
        n_quantiles=16,
    )


def _train_checkpoints(
    agent: Any,
    train_envs: Sequence[Any],
    eval_envs: Sequence[Any],
    episode_grid: Sequence[int],
    seed: int,
    label: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], List[float]]:
    random.seed(seed)
    np.random.seed(seed)
    grid = sorted({int(x) for x in episode_grid})
    grid_set = set(grid)
    max_episode = max(grid)
    rows: List[Dict[str, Any]] = []
    states: Dict[int, Dict[str, Any]] = {}
    train_rewards: List[float] = []
    for episode in range(1, max_episode + 1):
        env = random.choice(train_envs)
        state = env.reset()
        done = False
        total = 0.0
        step = 0
        while not done:
            requested = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(requested)
            executed = int(info.get("action", requested))
            agent.store(state, executed, reward, next_state, done)
            if step % UPDATE_FREQ == 0:
                agent.update()
            state = next_state
            total += float(reward)
            step += 1
        train_rewards.append(total)
        if episode in grid_set:
            result = exp9.evaluate_policy(
                eval_envs,
                lambda s, a=agent: a.select_action(s, training=False),
                n_runs=1,
            )
            states[episode] = agent.get_state()
            row = {"Seed": seed, "Episode": episode, **_metric_view(result), "epsilon": float(agent.epsilon)}
            rows.append(row)
            print(
                f"[{label}] seed={seed} episode={episode} reward={row['mean_reward']:.6f} "
                f"unmit={row['unmitigated_rate']:.6f} events={row['events_per_100_windows']:.6f}",
                flush=True,
            )
    return rows, states, train_rewards


def _save_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def run_reproduce(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    seed = int(args.seed)
    stage = out_dir / "a0_exp9_reproduction" / f"seed_{seed}"
    _, train, test, diagnostic, sources = _load_context("rms", args.fast, stage / "artifacts")
    train_envs = exp9.make_envs(train, seed=TRAIN_ENV_SEED)
    test_envs = exp9.make_envs(test, seed=TEST_ENV_SEED)
    agent = _make_exact_agent(train_envs[0].state_dim, seed, args.fast)
    rows, states, train_rewards = _train_checkpoints(agent, train_envs, test_envs, args.episode_grid, seed, "reproduce")
    best_row = max(rows, key=lambda row: float(row["mean_reward"]))
    best_episode = int(best_row["Episode"])
    agent.load_state(states[best_episode])
    final = exp9.evaluate_policy(test_envs, lambda s: agent.select_action(s, training=False), n_runs=1)
    _write_table(pd.DataFrame(rows), stage / "checkpoint_metrics.csv")
    _save_pickle(stage / "model.pkl", {
        "seed": seed,
        "state_dim": train_envs[0].state_dim,
        "best_episode": best_episode,
        "best_state": states[best_episode],
        "all_states": states,
    })
    summary = {
        "seed": seed,
        "legacy_test_selected_checkpoint": best_episode,
        "metrics": _metric_view(final),
        "risk_sources": sources,
        "diagnostic_report": diagnostic.report,
        "train_reward_count": len(train_rewards),
        "protocol_note": "Legacy test-checkpoint selection is retained only to diagnose Exp9 reproducibility.",
    }
    _write_json(stage / "summary.json", summary)
    return summary


def run_validation(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    seed = int(args.seed)
    stage = out_dir / "a1_validation" / f"seed_{seed}"
    train_core, val, val_labels, diagnostic, sources = _load_validation_context(
        args.fast,
        stage / "artifacts",
        args.val_ratio,
        args.split_seed,
    )
    train_envs = exp9.make_envs(train_core, seed=TRAIN_ENV_SEED)
    val_envs = exp9.make_envs(val, seed=TEST_ENV_SEED)
    agent = _make_exact_agent(train_envs[0].state_dim, seed, args.fast)
    rows, states, train_rewards = _train_checkpoints(agent, train_envs, val_envs, args.episode_grid, seed, "validation")
    _write_table(pd.DataFrame(rows), stage / "checkpoint_metrics.csv")
    _save_pickle(stage / "checkpoints.pkl", {
        "seed": seed,
        "state_dim": train_envs[0].state_dim,
        "states": states,
        "episode_grid": list(args.episode_grid),
    })
    summary = {
        "seed": seed,
        "train_scenarios": len(train_core),
        "validation_scenarios": len(val),
        "validation_groups": val_labels,
        "risk_sources": sources,
        "diagnostic_report": diagnostic.report,
        "train_reward_count": len(train_rewards),
    }
    _write_json(stage / "summary.json", summary)
    return summary


def _decision_action(agent: Any, state: np.ndarray, cfg: DecisionConfig, bias_scale: float) -> int:
    x = torch.as_tensor(agent.encode(state)[None, ...], dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        quantiles = agent.q_eval(x)[0]
    n_quantiles = int(quantiles.shape[1])
    keep = max(1, min(n_quantiles, int(math.ceil(cfg.cvar_alpha * n_quantiles))))
    sorted_q, _ = torch.sort(quantiles, dim=1)
    scores = sorted_q[:, :keep].mean(dim=1).clone()
    scale = float(bias_scale)
    scores[1:] += float(cfg.maintenance_bias) * scale
    scores[2] += float(cfg.urgency_bias) * scale
    scores[3] += 2.0 * float(cfg.urgency_bias) * scale
    return int(torch.argmax(scores).item())


def _collect_q_scale(agent: Any, envs: Sequence[Any], max_states: int = 384) -> float:
    spreads: List[float] = []
    for env in envs:
        state = env.reset()
        done = False
        while not done and len(spreads) < max_states:
            x = torch.as_tensor(agent.encode(state)[None, ...], dtype=torch.float32, device=agent.device)
            with torch.no_grad():
                means = agent.q_eval(x)[0].mean(dim=1).detach().cpu().numpy()
            spreads.append(float(np.max(means) - np.min(means)))
            action = agent.select_action(state, training=False)
            state, _, done, _ = env.step(action)
        if len(spreads) >= max_states:
            break
    return max(1e-3, float(np.median(spreads))) if spreads else 1.0


def _load_validation_agents(out_dir: Path, args: argparse.Namespace, episode: int):
    _, val, _, _, _ = _load_validation_context(
        args.fast,
        out_dir / "a2_selection" / "artifacts",
        args.val_ratio,
        args.split_seed,
    )
    val_envs = exp9.make_envs(val, seed=TEST_ENV_SEED)
    agents: Dict[int, Any] = {}
    for seed in args.seeds:
        payload = _load_pickle(out_dir / "a1_validation" / f"seed_{seed}" / "checkpoints.pkl")
        state = payload["states"][int(episode)]
        agent = _make_exact_agent(int(payload["state_dim"]), int(seed), args.fast)
        agent.load_state(state)
        agents[int(seed)] = agent
    return agents, val_envs


def _evaluate_config(
    agents: Dict[int, Any],
    envs: Sequence[Any],
    cfg: DecisionConfig,
    bias_scale: float,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    seed_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, float]] = []
    for seed, agent in sorted(agents.items()):
        result = exp9.evaluate_policy(
            envs,
            lambda s, a=agent, c=cfg: _decision_action(a, s, c, bias_scale),
            n_runs=1,
        )
        metrics = _metric_view(result)
        metric_rows.append(metrics)
        seed_rows.append({"Seed": seed, "Config": cfg.key, **metrics})
    return _aggregate_metric_views(metric_rows), seed_rows


def _profile_assessment(
    profile: str,
    metrics: Dict[str, float],
    baseline: Dict[str, float],
    incumbent: Dict[str, float],
) -> Dict[str, Any]:
    reward = metrics["mean_reward"]
    unmit = metrics["unmitigated_rate"]
    events = metrics["events_per_100_windows"]
    if profile == "safety":
        constraints = {
            "reward_noninferior": reward >= baseline["mean_reward"] - 0.25,
            "event_cap": events <= baseline["events_per_100_windows"] + 0.30,
        }
        improvement = incumbent["unmitigated_rate"] - unmit
        objective_met = improvement >= 0.001
        score = -100.0 * unmit - 2.0 * max(0.0, baseline["mean_reward"] - 0.25 - reward)
    elif profile == "balanced":
        constraints = {
            "safety_guard": unmit <= baseline["unmitigated_rate"] + 0.001,
            "event_cap": events <= baseline["events_per_100_windows"] + 0.20,
        }
        improvement = reward - incumbent["mean_reward"]
        objective_met = improvement >= 0.05
        score = reward - 100.0 * max(0.0, unmit - baseline["unmitigated_rate"] - 0.001)
    else:
        constraints = {
            "safety_guard": unmit <= baseline["unmitigated_rate"] + 0.002,
            "reward_noninferior": reward >= baseline["mean_reward"] - 0.25,
        }
        improvement = incumbent["events_per_100_windows"] - events
        objective_met = improvement >= 0.10
        score = -events - 2.0 * max(0.0, baseline["mean_reward"] - 0.25 - reward)
    return {
        "constraints": constraints,
        "feasible": bool(all(constraints.values())),
        "objective_improvement": float(improvement),
        "objective_met": bool(objective_met),
        "guarded_score": float(score),
    }


def _candidate_grid(incumbent: DecisionConfig, round_index: int, fast: bool = False) -> List[DecisionConfig]:
    alpha_levels = [0.25, 0.50, 0.75, 1.00]
    if fast:
        values = {
            incumbent,
            DecisionConfig(1.0, 0.0, 0.0),
            DecisionConfig(0.5, 0.0, 0.0),
            DecisionConfig(1.0, 0.5, 0.0),
            DecisionConfig(1.0, -0.5, 0.0),
        }
        return sorted(values, key=lambda c: (c.cvar_alpha, c.maintenance_bias, c.urgency_bias))
    if round_index == 1:
        alphas = alpha_levels
        maint = [-1.0, -0.5, 0.0, 0.5, 1.0]
        urgency = [-0.5, 0.0, 0.5]
    else:
        nearest = min(range(len(alpha_levels)), key=lambda i: abs(alpha_levels[i] - incumbent.cvar_alpha))
        alphas = [alpha_levels[i] for i in range(max(0, nearest - 1), min(len(alpha_levels), nearest + 2))]
        maint = [float(np.clip(incumbent.maintenance_bias + d, -1.5, 1.5)) for d in (-0.5, -0.25, 0.0, 0.25, 0.5)]
        urgency = [float(np.clip(incumbent.urgency_bias + d, -1.0, 1.0)) for d in (-0.25, 0.0, 0.25)]
    values = {DecisionConfig(float(a), float(m), float(u)) for a in alphas for m in maint for u in urgency}
    values.add(incumbent)
    return sorted(values, key=lambda c: (c.cvar_alpha, c.maintenance_bias, c.urgency_bias))


def _agent_choose_candidate(
    profile: PreferenceProfile,
    round_index: int,
    incumbent: DecisionConfig,
    shortlist: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    raw_dir: Path,
) -> Tuple[str, Dict[str, Any]]:
    prompt = {
        "task": "Select one bounded QR-DQN decision configuration from the supplied feasible shortlist.",
        "profile": asdict(profile),
        "round": round_index,
        "incumbent": asdict(incumbent),
        "parameter_meaning": {
            "cvar_alpha": "Fraction of lower return quantiles used for action selection; smaller values are more downside-risk averse.",
            "maintenance_bias": "Dimensionless bias applied to all maintenance actions relative to the median QR-DQN action gap.",
            "urgency_bias": "Additional dimensionless bias for preventive and immediate actions.",
        },
        "protocol_guards": [
            "You may select only a candidate_id present in the shortlist.",
            "Reward, state space, action space, network weights and test set cannot be changed.",
            "The program independently rejects candidates that violate constraints or fail to improve the role objective.",
        ],
        "shortlist": list(shortlist),
        "required_output": {
            "candidate_id": "exact candidate_id from shortlist",
            "rationale": "brief evidence-based trade-off explanation",
            "expected_operational_effect": "expected effect on safety, maintenance events and reward",
        },
    }
    if args.dry_run_agent:
        chosen = str(shortlist[0]["candidate_id"])
        return chosen, {"prompt": prompt, "parsed": {"candidate_id": chosen, "rationale": "deterministic smoke selection"}}
    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SF_API_KEY")
    if not api_key:
        raise RuntimeError("Formal Agent selection requires SILICONFLOW_API_KEY or SF_API_KEY.")
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            content, raw, latency = util._call_chat_completion(
                model_id=args.agent_model,
                prompt=json.dumps(prompt, ensure_ascii=False, indent=2),
                api_key=api_key,
                endpoint=args.llm_endpoint,
                temperature=0.05,
                max_tokens=900,
                timeout=args.llm_timeout,
            )
            parsed = util._extract_json_object(content)
            candidate_id = str(parsed.get("candidate_id", ""))
            allowed = {str(item["candidate_id"]) for item in shortlist}
            if candidate_id not in allowed:
                raise ValueError(f"Agent selected an unavailable candidate: {candidate_id}")
            record = {"attempt": attempt, "prompt": prompt, "content": content, "raw_response": raw, "latency_sec": latency, "parsed": parsed}
            _write_json(raw_dir / f"{profile.key}_round{round_index}_attempt{attempt}.json", record)
            return candidate_id, record
        except Exception as exc:
            last_error = exc
            _write_json(raw_dir / f"{profile.key}_round{round_index}_attempt{attempt}_error.json", {"prompt": prompt, "error": repr(exc)})
            time.sleep(min(5 * attempt, 15))
    raise RuntimeError(f"Agent selection failed for {profile.key} round {round_index}: {last_error}")


def run_select(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    stage = out_dir / "a2_selection"
    checkpoint_frames = []
    for seed in args.seeds:
        frame = pd.read_csv(out_dir / "a1_validation" / f"seed_{seed}" / "checkpoint_metrics.csv", encoding="utf-8-sig")
        checkpoint_frames.append(frame)
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    episode_summary = checkpoints.groupby("Episode", as_index=False).agg(
        mean_reward=("mean_reward", "mean"),
        reward_seed_std=("mean_reward", "std"),
        unmitigated_rate=("unmitigated_rate", "mean"),
        events_per_100=("events_per_100_windows", "mean"),
    )
    selected_episode = int(episode_summary.sort_values(["mean_reward", "Episode"], ascending=[False, True]).iloc[0]["Episode"])
    _write_table(episode_summary, stage / "episode_selection.csv")
    agents, val_envs = _load_validation_agents(out_dir, args, selected_episode)
    q_scales = [_collect_q_scale(agent, val_envs) for agent in agents.values()]
    bias_scale = float(np.median(q_scales))
    baseline_cfg = DecisionConfig()
    baseline_metrics, baseline_seed_rows = _evaluate_config(agents, val_envs, baseline_cfg, bias_scale)
    _write_table(pd.DataFrame(baseline_seed_rows), stage / "baseline_by_seed.csv")

    cache: Dict[str, Tuple[Dict[str, float], List[Dict[str, Any]]]] = {baseline_cfg.key: (baseline_metrics, baseline_seed_rows)}
    final_configs: Dict[str, DecisionConfig] = {"baseline": baseline_cfg}
    all_candidate_rows: List[Dict[str, Any]] = []
    round_records: List[Dict[str, Any]] = []
    raw_dir = stage / "agent_raw"

    for profile_key, profile in PROFILES.items():
        incumbent = baseline_cfg
        incumbent_metrics = baseline_metrics
        for round_index in (1, 2, 3):
            evaluated: List[Dict[str, Any]] = []
            for cfg in _candidate_grid(incumbent, round_index, args.fast):
                if cfg.key not in cache:
                    cache[cfg.key] = _evaluate_config(agents, val_envs, cfg, bias_scale)
                metrics, seed_rows = cache[cfg.key]
                assessment = _profile_assessment(profile_key, metrics, baseline_metrics, incumbent_metrics)
                row = {
                    "profile": profile_key,
                    "round": round_index,
                    "candidate_id": cfg.key,
                    **asdict(cfg),
                    **metrics,
                    **assessment,
                }
                evaluated.append(row)
                all_candidate_rows.append({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v) for k, v in row.items()})
            eligible = [row for row in evaluated if row["feasible"] and row["objective_met"]]
            eligible.sort(key=lambda row: float(row["guarded_score"]), reverse=True)
            if not eligible:
                round_records.append({
                    "profile": profile_key,
                    "round": round_index,
                    "status": "retain_no_feasible_improvement",
                    "incumbent": asdict(incumbent),
                    "incumbent_metrics": incumbent_metrics,
                })
                continue
            shortlist = []
            for row in eligible[:8]:
                shortlist.append({
                    "candidate_id": row["candidate_id"],
                    "parameters": {key: row[key] for key in ("cvar_alpha", "maintenance_bias", "urgency_bias")},
                    "mean_reward": row["mean_reward"],
                    "unmitigated_rate": row["unmitigated_rate"],
                    "events_per_100_windows": row["events_per_100_windows"],
                    "objective_improvement": row["objective_improvement"],
                    "guarded_score": row["guarded_score"],
                })
            chosen_id, agent_record = _agent_choose_candidate(profile, round_index, incumbent, shortlist, args, raw_dir)
            chosen_row = next(row for row in eligible if row["candidate_id"] == chosen_id)
            accepted = bool(chosen_row["feasible"] and chosen_row["objective_met"])
            previous = incumbent
            if accepted:
                incumbent = DecisionConfig(
                    float(chosen_row["cvar_alpha"]),
                    float(chosen_row["maintenance_bias"]),
                    float(chosen_row["urgency_bias"]),
                )
                incumbent_metrics = cache[incumbent.key][0]
            round_records.append({
                "profile": profile_key,
                "round": round_index,
                "status": "accepted" if accepted else "rejected",
                "previous": asdict(previous),
                "selected": asdict(incumbent),
                "selected_metrics": incumbent_metrics,
                "agent_parsed": agent_record.get("parsed", {}),
                "shortlist_ids": [item["candidate_id"] for item in shortlist],
            })
        final_configs[profile_key] = incumbent

    _write_table(pd.DataFrame(all_candidate_rows), stage / "candidate_evaluations.csv")
    _write_json(stage / "agent_round_trace.json", round_records)
    selection = {
        "selected_episode": selected_episode,
        "episode_selection_rule": "maximum three-seed mean validation reward; held-out test not accessed",
        "bias_scale": bias_scale,
        "q_scale_by_seed": {str(seed): value for seed, value in zip(sorted(agents), q_scales)},
        "baseline_validation_metrics": baseline_metrics,
        "selected_configs": {key: asdict(cfg) for key, cfg in final_configs.items()},
        "profiles": {key: asdict(value) for key, value in PROFILES.items()},
        "rounds": round_records,
    }
    _write_json(stage / "selection.json", selection)
    return selection


def _train_fixed_budget(agent: Any, train_envs: Sequence[Any], episodes: int, seed: int) -> List[float]:
    random.seed(seed)
    np.random.seed(seed)
    rewards: List[float] = []
    for episode in range(1, episodes + 1):
        env = random.choice(train_envs)
        state = env.reset()
        done = False
        total = 0.0
        step = 0
        while not done:
            requested = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(requested)
            agent.store(state, int(info.get("action", requested)), reward, next_state, done)
            if step % UPDATE_FREQ == 0:
                agent.update()
            state = next_state
            total += float(reward)
            step += 1
        rewards.append(total)
        if episode % max(1, episodes // 10) == 0 or episode == episodes:
            print(f"[final-train] seed={seed} episode={episode}/{episodes} epsilon={agent.epsilon:.4f}", flush=True)
    return rewards


def run_final(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    seed = int(args.seed)
    stage = out_dir / "a3_final_test" / f"seed_{seed}"
    selection = _read_json(out_dir / "a2_selection" / "selection.json")
    episodes = int(selection["selected_episode"])
    bias_scale = float(selection["bias_scale"])
    configs = {key: DecisionConfig(**value) for key, value in selection["selected_configs"].items()}
    _, train_full, test, diagnostic, sources = _load_context("rms", args.fast, stage / "artifacts")
    train_envs = exp9.make_envs(train_full, seed=TRAIN_ENV_SEED)
    test_envs = exp9.make_envs(test, seed=TEST_ENV_SEED)
    agent = _make_exact_agent(train_envs[0].state_dim, seed, args.fast)
    train_rewards = _train_fixed_budget(agent, train_envs, episodes, seed)
    rows: List[Dict[str, Any]] = []
    for track, cfg in configs.items():
        result = exp9.evaluate_policy(
            test_envs,
            lambda s, a=agent, c=cfg: _decision_action(a, s, c, bias_scale),
            n_runs=1,
        )
        rows.append({"Track": track, "Seed": seed, **asdict(cfg), **_metric_view(result)})
    _write_table(pd.DataFrame(rows), stage / "test_metrics.csv")
    _save_pickle(stage / "model.pkl", {
        "seed": seed,
        "state_dim": train_envs[0].state_dim,
        "episodes": episodes,
        "state": agent.get_state(),
        "bias_scale": bias_scale,
        "configs": {key: asdict(value) for key, value in configs.items()},
    })
    summary = {
        "seed": seed,
        "training_episodes_selected_on_validation": episodes,
        "test_evaluations_per_track": 1,
        "risk_sources": sources,
        "diagnostic_report": diagnostic.report,
        "train_reward_count": len(train_rewards),
    }
    _write_json(stage / "summary.json", summary)
    return summary


def run_merge(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    repro_rows = []
    for seed in args.seeds:
        summary = _read_json(out_dir / "a0_exp9_reproduction" / f"seed_{seed}" / "summary.json")
        repro_rows.append({"Seed": seed, "Best episode": summary["legacy_test_selected_checkpoint"], **summary["metrics"]})
    repro_df = pd.DataFrame(repro_rows)
    repro_mean = _aggregate_metric_views([{key: float(row[key]) for key in _metric_view({}).keys()} for row in repro_rows])
    historical_path = THIS_DIR / "outputs" / "exp9_qrdqn_v6_model_comparison_parallel_merged" / "exp9_v6_parallel_merged_table.csv"
    historical: Dict[str, float] = {}
    if historical_path.exists():
        frame = pd.read_csv(historical_path, encoding="utf-8-sig")
        row = frame[frame["Method"] == "QR-DQN"].iloc[0]
        historical = {
            "mean_reward": float(row["Avg. reward"]),
            "unmitigated_rate": float(row["Unmitigated rate"]),
            "maintenance_rate": float(row["Intervention event rate"]),
            "downtime_cost": float(row["Downtime cost"]),
        }
    _write_table(repro_df, out_dir / "a0_exp9_reproduction" / "reproduction_by_seed.csv")

    final_frames = [
        pd.read_csv(out_dir / "a3_final_test" / f"seed_{seed}" / "test_metrics.csv", encoding="utf-8-sig")
        for seed in args.seeds
    ]
    final_all = pd.concat(final_frames, ignore_index=True)
    numeric_cols = [c for c in final_all.columns if c not in {"Track", "Seed"}]
    mean_rows = []
    for track, group in final_all.groupby("Track", sort=False):
        row: Dict[str, Any] = {"Track": track, "Seeds": len(group)}
        for col in numeric_cols:
            row[col] = float(group[col].astype(float).mean())
            row[f"{col}_seed_std"] = float(group[col].astype(float).std(ddof=0))
        mean_rows.append(row)
    final_mean = pd.DataFrame(mean_rows)
    _write_table(final_all, out_dir / "a3_final_test" / "test_metrics_by_seed.csv")
    _write_table(final_mean, out_dir / "a3_final_test" / "test_metrics_aggregate.csv")
    bridge = {
        "historical_exp9_path": str(historical_path),
        "historical_exp9": historical,
        "rms_reproduction_mean": repro_mean,
        "deltas_reproduction_minus_historical": {
            key: float(repro_mean[key] - historical[key]) for key in historical if key in repro_mean
        },
        "confirmed_protocol_alignment": {
            "exp9_20260701": {"risk_sources": {"rms_fallback": 144}},
            "exp10_v6_a_b": {"risk_sources": {"rms_fallback": 144}},
            "exp10_c_only": "Exp7 DSHI is reserved for structured multi-agent explanatory evidence.",
        },
    }
    _write_json(out_dir / "a0_exp9_reproduction" / "exp9_bridge_report.json", bridge)
    summary = {
        "exp9_bridge_report": str(out_dir / "a0_exp9_reproduction" / "exp9_bridge_report.json"),
        "final_table": str(out_dir / "a3_final_test" / "test_metrics_aggregate.csv"),
    }
    _write_json(out_dir / "a3_final_test" / "merge_summary.json", summary)
    return summary


def _risk_bin(state: np.ndarray) -> str:
    priority = max(
        float(state[exp9.IDX_CURRENT_RISK] * 3.0),
        float(state[exp9.IDX_FORECAST_RISK] * 3.0),
    )
    return "low" if priority < 0.7 else ("medium" if priority < 1.0 else "high")


def _reservoir_add(pool: List[Any], item: Any, seen: int, cap: int, rng: np.random.Generator) -> None:
    if len(pool) < cap:
        pool.append(item)
    else:
        index = int(rng.integers(0, seen))
        if index < cap:
            pool[index] = item


def _collect_eligible_snapshots(
    envs: Sequence[Any],
    policy: Any,
    seed: int,
    per_stratum: int,
) -> Tuple[Dict[Tuple[int, str], List[Any]], Dict[Tuple[int, str], int], Dict[str, int]]:
    rng = np.random.default_rng(20260713 + seed)
    pools: Dict[Tuple[int, str], List[Any]] = {}
    counts: Dict[Tuple[int, str], int] = {}
    totals = {"all_states": 0, "eligible_states": 0, "cooldown_states": 0, "requested_executed_mismatch": 0}
    for env in envs:
        state = env.reset()
        done = False
        while not done:
            totals["all_states"] += 1
            requested = int(policy(state))
            eligible = int(env.action_cooldown_left) == 0
            if eligible:
                totals["eligible_states"] += 1
                key = (requested, _risk_bin(state))
                counts[key] = counts.get(key, 0) + 1
                pools.setdefault(key, [])
                item = (copy.deepcopy(env), np.asarray(state, dtype=np.float32).copy(), requested)
                _reservoir_add(pools[key], item, counts[key], per_stratum, rng)
            else:
                totals["cooldown_states"] += 1
            state, _, done, info = env.step(requested)
            if int(info.get("action", requested)) != requested:
                totals["requested_executed_mismatch"] += 1
    return pools, counts, totals


def _counterfactual_rollout(env: Any, first_action: int, horizon: int, policy: Any) -> Dict[str, Any]:
    trial = copy.deepcopy(env)
    state, reward, done, info = trial.step(first_action)
    first_executed = int(info.get("action", first_action))
    discounted_return = float(reward)
    unmitigated = int(bool(info.get("unmitigated", False)))
    downtime = float(info.get("downtime_cost", 0.0))
    steps = 1
    while not done and steps < horizon:
        requested = int(policy(state))
        state, reward, done, info = trial.step(requested)
        discounted_return += float(reward) * (0.95 ** steps)
        unmitigated += int(bool(info.get("unmitigated", False)))
        downtime += float(info.get("downtime_cost", 0.0))
        steps += 1
    return {
        "return": discounted_return,
        "first_executed_action": first_executed,
        "unmitigated_steps": unmitigated,
        "downtime_cost": downtime,
        "simulated_steps": steps,
    }


def run_audit(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    seed = int(args.seed)
    stage = out_dir / "b_counterfactual_audit" / f"seed_{seed}"
    model_payload = _load_pickle(out_dir / "a3_final_test" / f"seed_{seed}" / "model.pkl")
    _, _, test, diagnostic, sources = _load_context("rms", args.fast, stage / "artifacts")
    base_envs = exp9.make_envs(test, seed=TEST_ENV_SEED)
    agent = _make_exact_agent(int(model_payload["state_dim"]), seed, args.fast)
    agent.load_state(model_payload["state"])
    bias_scale = float(model_payload["bias_scale"])
    configs = {key: DecisionConfig(**value) for key, value in model_payload["configs"].items()}
    if args.fast:
        configs = {"baseline": configs["baseline"]}
    horizons = args.audit_horizons
    case_rows: List[Dict[str, Any]] = []
    availability_rows: List[Dict[str, Any]] = []
    case_index = 0
    for track, cfg in configs.items():
        policy = lambda s, a=agent, c=cfg: _decision_action(a, s, c, bias_scale)
        pools, population, totals = _collect_eligible_snapshots(
            base_envs,
            policy,
            seed,
            args.audit_per_stratum,
        )
        for action in range(exp9.N_ACTIONS):
            for risk_bin in ("low", "medium", "high"):
                key = (action, risk_bin)
                availability_rows.append({
                    "Track": track,
                    "Seed": seed,
                    "Action": exp9.ACTION_NAMES[action],
                    "Risk bin": risk_bin,
                    "Population eligible states": population.get(key, 0),
                    "Sampled states": len(pools.get(key, [])),
                    **totals,
                })
        for (requested_action, risk_bin), samples in pools.items():
            pop_count = population[(requested_action, risk_bin)]
            sample_weight = float(pop_count / max(len(samples), 1))
            for snapshot_env, state, requested in samples:
                case_index += 1
                for horizon in horizons:
                    scores = {
                        action: _counterfactual_rollout(snapshot_env, action, horizon, policy)
                        for action in range(exp9.N_ACTIONS)
                    }
                    if any(item["first_executed_action"] != action for action, item in scores.items()):
                        raise RuntimeError("Ineligible cooldown state leaked into the audit sample.")
                    best_return = max(item["return"] for item in scores.values())
                    tolerance = max(float(args.tie_tolerance), 0.005 * abs(best_return))
                    reference_set = [action for action, item in scores.items() if best_return - item["return"] <= tolerance]
                    policy_return = scores[requested]["return"]
                    case_rows.append({
                        "Case": f"seed{seed}_{track}_{case_index:05d}",
                        "Track": track,
                        "Seed": seed,
                        "Horizon": horizon,
                        "Risk bin": risk_bin,
                        "Requested action": exp9.ACTION_NAMES[requested],
                        "Executed action": exp9.ACTION_NAMES[requested],
                        "Reference action set": "/".join(exp9.ACTION_NAMES[a] for a in reference_set),
                        "Agreement": int(requested in reference_set),
                        "Regret": max(0.0, best_return - policy_return),
                        "Tie tolerance": tolerance,
                        "Population weight": sample_weight,
                        "Current risk": float(state[exp9.IDX_CURRENT_RISK] * 3.0),
                        "Forecast risk": float(state[exp9.IDX_FORECAST_RISK] * 3.0),
                        "Diagnostic confidence": float(state[exp9.IDX_DIAG_CONF]),
                        "Cooldown before action": int(snapshot_env.action_cooldown_left),
                        "Scenario private": snapshot_env.scenario.name,
                        "Step private": int(snapshot_env.t),
                        "Counterfactual scores": json.dumps({exp9.ACTION_NAMES[a]: v for a, v in scores.items()}, ensure_ascii=False),
                    })
    cases = pd.DataFrame(case_rows)
    availability = pd.DataFrame(availability_rows)
    _write_table(cases, stage / "audit_cases.csv")
    _write_table(availability, stage / "action_availability.csv")
    summary = {
        "seed": seed,
        "tracks": list(configs),
        "horizons": horizons,
        "sampled_case_horizon_rows": len(cases),
        "risk_sources": sources,
        "diagnostic_report": diagnostic.report,
        "decision_eligibility_rule": "action_cooldown_left == 0 before the requested action",
    }
    _write_json(stage / "summary.json", summary)
    return summary


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.astype(float).to_numpy()
    x = values.astype(float).to_numpy()
    return float(np.sum(w * x) / max(np.sum(w), 1e-12))


def run_audit_merge(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    root = out_dir / "b_counterfactual_audit"
    cases = pd.concat([
        pd.read_csv(root / f"seed_{seed}" / "audit_cases.csv", encoding="utf-8-sig")
        for seed in args.seeds
    ], ignore_index=True)
    availability = pd.concat([
        pd.read_csv(root / f"seed_{seed}" / "action_availability.csv", encoding="utf-8-sig")
        for seed in args.seeds
    ], ignore_index=True)
    _write_table(cases, root / "audit_cases_all_seeds.csv")
    _write_table(availability, root / "action_availability_all_seeds.csv")
    rows = []
    for (track, horizon, action), group in cases.groupby(["Track", "Horizon", "Executed action"], sort=False):
        rows.append({
            "Track": track,
            "Horizon": int(horizon),
            "Executed action": action,
            "Sampled cases": len(group),
            "Unweighted agreement": float(group["Agreement"].mean()),
            "Weighted agreement": _weighted_mean(group["Agreement"], group["Population weight"]),
            "Unweighted mean regret": float(group["Regret"].mean()),
            "Weighted mean regret": _weighted_mean(group["Regret"], group["Population weight"]),
            "Median regret": float(group["Regret"].median()),
            "P90 regret": float(group["Regret"].quantile(0.90)),
        })
    per_action = pd.DataFrame(rows)
    overall_rows = []
    for (track, horizon), group in cases.groupby(["Track", "Horizon"], sort=False):
        overall_rows.append({
            "Track": track,
            "Horizon": int(horizon),
            "Sampled cases": len(group),
            "Weighted agreement": _weighted_mean(group["Agreement"], group["Population weight"]),
            "Weighted mean regret": _weighted_mean(group["Regret"], group["Population weight"]),
            "Median regret": float(group["Regret"].median()),
            "P90 regret": float(group["Regret"].quantile(0.90)),
        })
    overall = pd.DataFrame(overall_rows)
    _write_table(per_action, root / "audit_summary_by_action.csv")
    _write_table(overall, root / "audit_summary_overall.csv")
    integrity = {
        "cooldown_leaks": int((cases["Cooldown before action"].astype(int) != 0).sum()),
        "requested_executed_mismatches": int((cases["Requested action"] != cases["Executed action"]).sum()),
        "seeds": list(args.seeds),
        "horizons": list(args.audit_horizons),
        "weighting": "population count divided by sampled count within seed/track/action/risk-bin stratum",
        "reference_name": "finite-horizon counterfactual rollout reference, not a global oracle",
    }
    _write_json(root / "audit_integrity.json", integrity)
    return {"overall_table": str(root / "audit_summary_overall.csv"), "integrity": integrity}


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp10-v6 guarded preference selection and audit.")
    parser.add_argument("--phase", choices=["reproduce", "validation", "select", "final", "merge", "audit", "audit-merge"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--episode-grid", default="100,300,500,1000")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=20260713)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dry-run-agent", action="store_true")
    parser.add_argument("--agent-model", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--llm-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--audit-horizons", default="8,16,32")
    parser.add_argument("--audit-per-stratum", type=int, default=3)
    parser.add_argument("--tie-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    args.seeds = _parse_ints(args.seeds)
    args.episode_grid = _parse_ints(args.episode_grid)
    args.audit_horizons = _parse_ints(args.audit_horizons)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "exp10_v6_run_manifest.json"
    manifest = _read_json(config_path) if config_path.exists() else {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "risk_protocol": {
            "exp10_a_b": "formal_exp9_rms_fallback",
            "expected_formal_sources": {"rms_fallback": 144},
            "exp7_dshi_usage": "exp10_c_structured_explanatory_evidence_only",
        },
        "invocations": [],
    }
    manifest["invocations"].append({"time": dt.datetime.now().isoformat(timespec="seconds"), "arguments": vars(args)})
    _write_json(config_path, manifest)
    functions = {
        "reproduce": run_reproduce,
        "validation": run_validation,
        "select": run_select,
        "final": run_final,
        "merge": run_merge,
        "audit": run_audit,
        "audit-merge": run_audit_merge,
    }
    result = functions[args.phase](args, out_dir)
    completion = out_dir / f"completion_{args.phase}_seed{args.seed}.json"
    _write_json(completion, {"completed_at": dt.datetime.now().isoformat(timespec="seconds"), "phase": args.phase, "seed": args.seed, "result": result})
    print(f"[complete] phase={args.phase} seed={args.seed} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
