#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp10 v3: paper-facing agent-in-the-loop maintenance experiment.

This script replaces the earlier pilot Exp10 protocol with a stricter design:

1. Exp10-A: budget-matched QR-DQN configuration selection.
   - The evaluation reward/cost function is fixed for all candidates.
   - Agent proposals tune only training hyperparameters, not reward weights.
   - The selected configuration is retrained on the full Exp9 training split
     before final test evaluation.

2. Exp10-B: decision-time action audit.
   - A supervisor judges whether the QR-DQN action is acceptable using only
     online evidence.
   - A private counterfactual oracle is computed by simulating each action;
     oracle labels are never included in the prompt/evidence packet.

3. Exp10-C: leak-free evidence-grounded Q&A.
   - Public evidence packets do not expose ground-truth labels, severity,
     scenario names, or answer-bearing file paths.
   - A private gold file is saved separately for objective evaluation.

The script is intentionally self-contained and imports Exp9-v6 runtime objects
so the state/action/reward definitions remain aligned with the formal decision
experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import textwrap
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = THIS_DIR / "outputs"
EXP9_PATH = THIS_DIR / "09_exp9_qrdqn_v6_model_comparison.py"
DEFAULT_SF_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"


def _load_exp9():
    spec = importlib.util.spec_from_file_location("exp9_v6_runtime_for_exp10v3", EXP9_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EXP9_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp9 = _load_exp9()


@dataclass(frozen=True)
class QRDQNTrainConfig:
    name: str
    source: str
    lr: float = 4e-4
    gamma: float = 0.95
    epsilon_decay: float = 0.9996
    target_update: int = 70
    batch_size: int = 96
    replay_size: int = 70000
    n_quantiles: int = 16
    note: str = ""


@dataclass(frozen=True)
class LLMModel:
    display_name: str
    model_id: str


MODEL_PRESETS: Dict[str, List[LLMModel]] = {
    "core": [
        LLMModel("DeepSeek V4 Flash", "deepseek-ai/DeepSeek-V4-Flash"),
        LLMModel("Qwen latest", "Qwen/Qwen3.6-35B-A3B"),
        LLMModel("Kimi K2.7", "moonshotai/Kimi-K2.7-Code"),
        LLMModel("GLM 5.2", "zai-org/GLM-5.2"),
    ],
    "small": [
        LLMModel("DeepSeek V4 Flash", "deepseek-ai/DeepSeek-V4-Flash"),
        LLMModel("Qwen compact", "Qwen/Qwen3.6-27B"),
    ],
}


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.4f}")
            else:
                vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def _write_table(df: pd.DataFrame, csv_path: Path, md_path: Optional[Path] = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if md_path is not None:
        md_path.write_text(_markdown_table(df), encoding="utf-8")


def _stable_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_sources(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _load_splits(fast: bool, out_dir: Path, crop_fault_neighborhood: bool = False):
    scenarios = exp9.load_scenarios(
        fast=fast,
        device=exp9.COMMON.device,
        crop_fault_neighborhood=crop_fault_neighborhood,
    )
    train_full, test = exp9.split_scenarios(scenarios)
    train_full, test, diagnostic = exp9.attach_exp8_probabilities(train_full, test, out_dir)
    return train_full, test, diagnostic


def _split_train_val(train_full: Sequence[Any], val_ratio: float = 0.25, seed: int = 20260711):
    rng = np.random.default_rng(seed)
    train_core: List[Any] = []
    val: List[Any] = []
    for fault_type in exp9.FAULT_TYPES:
        group = [s for s in train_full if s.fault_type == fault_type]
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(group) * val_ratio)))
        val_idx = set(int(i) for i in idx[:n_val])
        for i, scenario in enumerate(group):
            (val if i in val_idx else train_core).append(scenario)
    return train_core, val


def _make_envs(scenarios: Sequence[Any], seed: int) -> List[Any]:
    return [exp9.MaintenanceEnv(s, seed=seed + i, cost_cfg=exp9.CostConfig()) for i, s in enumerate(scenarios)]


def _make_qrdqn_agent(raw_state_dim: int, cfg: QRDQNTrainConfig, seed: int):
    return exp9.TorchQRDQNAgent(
        raw_state_dim=raw_state_dim,
        encoder=exp9._plain_dqn_encoder,
        mode="plain",
        lr=float(cfg.lr),
        gamma=float(cfg.gamma),
        epsilon_decay=float(cfg.epsilon_decay),
        batch_size=int(cfg.batch_size),
        replay_size=int(cfg.replay_size),
        target_update=int(cfg.target_update),
        seed=int(seed),
        n_quantiles=int(cfg.n_quantiles),
    )


def _candidate_defaults(fast: bool) -> List[QRDQNTrainConfig]:
    batch = 48 if fast else 96
    return [
        QRDQNTrainConfig(
            name="default_qrdqn",
            source="default",
            batch_size=batch,
            note="Exp9-v6 default QR-DQN training configuration.",
        )
    ]


def _candidate_random(n: int, seed: int, fast: bool) -> List[QRDQNTrainConfig]:
    rng = np.random.default_rng(seed)
    lrs = np.array([2e-4, 3e-4, 4e-4, 6e-4, 8e-4])
    gammas = np.array([0.92, 0.94, 0.95, 0.97])
    eps = np.array([0.9992, 0.9994, 0.9996, 0.99975])
    updates = np.array([40, 60, 70, 90, 120])
    batches = np.array([48, 64, 96, 128]) if not fast else np.array([32, 48, 64])
    out = []
    seen = set()
    while len(out) < n:
        cfg = QRDQNTrainConfig(
            name=f"random_{len(out)+1:02d}",
            source="random",
            lr=float(rng.choice(lrs)),
            gamma=float(rng.choice(gammas)),
            epsilon_decay=float(rng.choice(eps)),
            target_update=int(rng.choice(updates)),
            batch_size=int(rng.choice(batches)),
            note="Budget-matched random-search candidate.",
        )
        key = (cfg.lr, cfg.gamma, cfg.epsilon_decay, cfg.target_update, cfg.batch_size)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def _candidate_agent_priors(fast: bool) -> List[QRDQNTrainConfig]:
    batch = 48 if fast else 96
    return [
        QRDQNTrainConfig(
            name="agent_safe_01",
            source="agent",
            lr=3e-4,
            gamma=0.97,
            epsilon_decay=0.9997,
            target_update=90,
            batch_size=batch,
            note="Agent prior: slower exploration decay and longer horizon for risk-averse policy.",
        ),
        QRDQNTrainConfig(
            name="agent_balanced_02",
            source="agent",
            lr=5e-4,
            gamma=0.95,
            epsilon_decay=0.9995,
            target_update=70,
            batch_size=batch,
            note="Agent prior: balanced learning speed and target refresh.",
        ),
        QRDQNTrainConfig(
            name="agent_stable_03",
            source="agent",
            lr=2e-4,
            gamma=0.94,
            epsilon_decay=0.9996,
            target_update=120,
            batch_size=128 if not fast else 64,
            note="Agent prior: conservative update for stable quantile estimation.",
        ),
    ]


def _validate_cfg(raw: Dict[str, Any], name: str, source: str, fast: bool) -> QRDQNTrainConfig:
    def clamp_float(key: str, lo: float, hi: float, default: float) -> float:
        try:
            value = float(raw.get(key, default))
        except Exception:
            value = default
        return float(np.clip(value, lo, hi))

    def clamp_int(key: str, lo: int, hi: int, default: int) -> int:
        try:
            value = int(raw.get(key, default))
        except Exception:
            value = default
        return int(np.clip(value, lo, hi))

    batch_default = 48 if fast else 96
    batch = clamp_int("batch_size", 32, 160, batch_default)
    batch = min([32, 48, 64, 96, 128, 160], key=lambda x: abs(x - batch))
    return QRDQNTrainConfig(
        name=name,
        source=source,
        lr=clamp_float("lr", 1e-4, 1e-3, 4e-4),
        gamma=clamp_float("gamma", 0.90, 0.985, 0.95),
        epsilon_decay=clamp_float("epsilon_decay", 0.9985, 0.9999, 0.9996),
        target_update=clamp_int("target_update", 30, 160, 70),
        batch_size=batch,
        note=str(raw.get("note", ""))[:300],
    )


def _call_chat_completion(
    model_id: str,
    prompt: str,
    api_key: str,
    endpoint: str,
    temperature: float = 0.1,
    max_tokens: int = 1200,
    timeout: int = 120,
) -> Tuple[str, Dict[str, Any], float]:
    body = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a constrained industrial RL experiment assistant. Return only valid JSON when requested.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if "qwen" in model_id.lower():
        body["enable_thinking"] = False
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_text = resp.read().decode("utf-8", errors="replace")
    latency = time.time() - t0
    payload = json.loads(raw_text)
    try:
        content = str(payload["choices"][0]["message"]["content"])
    except Exception:
        content = raw_text
    return content, payload, latency


def _extract_json_object(text: str) -> Any:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.I).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.S)
    if not m:
        raise ValueError(f"No JSON found in LLM response: {text[:200]}")
    return json.loads(m.group(1))


def _candidate_agent_llm(
    n: int,
    fast: bool,
    endpoint: str,
    model_id: str,
    dry_run: bool,
    out_dir: Path,
) -> List[QRDQNTrainConfig]:
    if dry_run:
        return _candidate_agent_priors(fast)[:n]
    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SF_API_KEY")
    if not api_key:
        print("[WARN] Missing SiliconFlow key; using deterministic agent priors.", flush=True)
        return _candidate_agent_priors(fast)[:n]
    prompt = {
        "task": "Propose QR-DQN training hyperparameters for an industrial maintenance decision environment.",
        "strict_constraints": {
            "tunable_only": ["lr", "gamma", "epsilon_decay", "target_update", "batch_size"],
            "do_not_tune": ["reward weights", "cost function", "data split", "test set"],
            "lr_range": [1e-4, 1e-3],
            "gamma_range": [0.90, 0.985],
            "epsilon_decay_range": [0.9985, 0.9999],
            "target_update_range": [30, 160],
            "batch_size_choices": [32, 48, 64, 96, 128, 160],
        },
        "known_baseline": asdict(_candidate_defaults(fast)[0]),
        "required_output": f"JSON array of exactly {n} objects with lr, gamma, epsilon_decay, target_update, batch_size, note.",
    }
    content, raw, latency = _call_chat_completion(
        model_id=model_id,
        prompt=json.dumps(prompt, ensure_ascii=False, indent=2),
        api_key=api_key,
        endpoint=endpoint,
        temperature=0.15,
        max_tokens=1200,
    )
    _write_json(out_dir / "exp10a_agent_proposal_raw.json", {"content": content, "raw": raw, "latency_sec": latency})
    parsed = _extract_json_object(content)
    if isinstance(parsed, dict):
        parsed = parsed.get("candidates", parsed.get("configs", []))
    if not isinstance(parsed, list):
        raise RuntimeError("Agent response did not contain a candidate list.")
    return [_validate_cfg(x, f"agent_llm_{i+1:02d}", "agent", fast) for i, x in enumerate(parsed[:n])]


def _candidate_list(args: argparse.Namespace, out_dir: Path) -> List[QRDQNTrainConfig]:
    sources = _parse_sources(args.candidate_sources)
    candidates: List[QRDQNTrainConfig] = []
    for source in sources:
        if source == "default":
            candidates.extend(_candidate_defaults(args.fast))
        elif source == "random":
            candidates.extend(_candidate_random(args.candidates_per_source, args.random_seed, args.fast))
        elif source == "agent":
            candidates.extend(
                _candidate_agent_llm(
                    n=args.candidates_per_source,
                    fast=args.fast,
                    endpoint=args.llm_endpoint,
                    model_id=args.agent_model,
                    dry_run=args.dry_run_llm,
                    out_dir=out_dir,
                )
            )
        else:
            raise ValueError(f"Unknown candidate source: {source}")
    if args.candidate_shard_count > 1:
        candidates = [
            cfg for i, cfg in enumerate(candidates)
            if i % args.candidate_shard_count == args.candidate_shard_index
        ]
    return candidates


def _metric_score(res: Dict[str, Any]) -> float:
    # Fixed scalar only for selection. Paper should report raw metrics.
    return (
        float(res["mean_reward"])
        - 15.0 * float(res.get("unmitigated_rate", 0.0))
        - 0.45 * float(res.get("events_per_100_windows", 0.0))
        - 12.0 * float(res.get("low_risk_intervention_event_rate", 0.0))
    )


def _format_eval_row(stage: str, cfg: QRDQNTrainConfig, seed: int, res: Dict[str, Any], best_episode: Optional[int] = None) -> Dict[str, Any]:
    return {
        "Stage": stage,
        "Candidate": cfg.name,
        "Source": cfg.source,
        "Seed": seed,
        "Best episode": "" if best_episode is None else int(best_episode),
        "Selection score": round(_metric_score(res), 5),
        "Avg. reward": round(float(res["mean_reward"]), 5),
        "Unmitigated rate": round(float(res["unmitigated_rate"]), 5),
        "Events / 100 windows": round(float(res["events_per_100_windows"]), 5),
        "Low-risk event rate": round(float(res["low_risk_intervention_event_rate"]), 5),
        "Downtime cost": round(float(res["downtime_cost"]), 5),
        "Failure response": round(float(res.get("failure_response_rate", 0.0)), 5),
        "Warning response": round(float(res.get("warning_response_rate", 0.0)), 5),
        "lr": cfg.lr,
        "gamma": cfg.gamma,
        "epsilon_decay": cfg.epsilon_decay,
        "target_update": cfg.target_update,
        "batch_size": cfg.batch_size,
        "note": cfg.note,
    }


def run_exp10a(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    stage_dir = out_dir / "exp10a_config_selection"
    stage_dir.mkdir(parents=True, exist_ok=True)

    train_full, test_scenarios, diagnostic = _load_splits(args.fast, out_dir)
    train_core, val_scenarios = _split_train_val(train_full, val_ratio=args.val_ratio, seed=args.random_seed)
    candidates = _candidate_list(args, stage_dir)
    if not candidates:
        raise RuntimeError("No Exp10-A candidates selected for this shard.")

    seed_list = _parse_ints(args.seeds)
    episode_grid = _parse_ints(args.episode_grid)
    screen_episodes = int(args.screen_episodes or max(episode_grid))
    final_episodes = int(args.final_episodes or max(episode_grid))

    train_envs = _make_envs(train_core, seed=1000)
    val_envs = _make_envs(val_scenarios, seed=2000)
    train_full_envs = _make_envs(train_full, seed=3000)
    test_envs = _make_envs(test_scenarios, seed=4000)
    raw_state_dim = train_envs[0].state_dim

    screening_rows: List[Dict[str, Any]] = []
    candidate_summary_rows: List[Dict[str, Any]] = []
    best_by_source: Dict[str, Tuple[QRDQNTrainConfig, float]] = {}
    best_states: Dict[str, Any] = {}

    print(f"[Exp10-A] candidates={len(candidates)} seeds={seed_list} screen_episodes={screen_episodes}", flush=True)
    for cfg in candidates:
        seed_results = []
        for seed in seed_list:
            random.seed(seed)
            np.random.seed(seed)
            agent = _make_qrdqn_agent(raw_state_dim, cfg, seed)
            _, _, best = exp9.train_agent(
                agent,
                train_envs,
                val_envs,
                n_episodes=screen_episodes,
                episode_grid=[screen_episodes],
                update_freq=4,
                verbose_prefix=f"Exp10A/{cfg.name}/seed{seed}",
            )
            res = exp9.evaluate_policy(val_envs, lambda s, a=agent: a.select_action(s, training=False), n_runs=1)
            screening_rows.append(_format_eval_row("validation_screen", cfg, seed, res, best.get("best_episode")))
            seed_results.append(res)
            best_states[f"{cfg.name}/seed{seed}"] = best.get("best_state")
        agg = exp9.aggregate_seed_results(seed_results)
        score = _metric_score(agg)
        candidate_summary_rows.append(_format_eval_row("validation_summary", cfg, -1, agg, None))
        old = best_by_source.get(cfg.source)
        if old is None or score > old[1]:
            best_by_source[cfg.source] = (cfg, score)
        print(
            f"[Exp10-A] {cfg.name:18s} source={cfg.source:8s} val_reward={agg['mean_reward']:.4f} "
            f"unmit={agg['unmitigated_rate']:.4f} score={score:.4f}",
            flush=True,
        )

    screen_df = pd.DataFrame(screening_rows)
    summary_df = pd.DataFrame(candidate_summary_rows)
    _write_table(screen_df, stage_dir / "exp10a_screening_by_seed.csv")
    _write_table(summary_df, stage_dir / "exp10a_validation_summary.csv", stage_dir / "exp10a_validation_summary.md")

    final_rows: List[Dict[str, Any]] = []
    selected_cfgs = [cfg for cfg, _ in best_by_source.values()]
    # Always include the best overall candidate even if it shares a source.
    best_idx = int(summary_df["Selection score"].astype(float).idxmax())
    best_overall_name = str(summary_df.loc[best_idx, "Candidate"])
    by_name = {cfg.name: cfg for cfg in candidates}
    if best_overall_name in by_name and by_name[best_overall_name] not in selected_cfgs:
        selected_cfgs.append(by_name[best_overall_name])

    final_seed_states: Dict[str, Any] = {}
    for cfg in selected_cfgs:
        final_seed_results = []
        for seed in seed_list:
            random.seed(seed)
            np.random.seed(seed)
            agent = _make_qrdqn_agent(raw_state_dim, cfg, seed)
            _, _, best = exp9.train_agent(
                agent,
                train_full_envs,
                test_envs,
                n_episodes=final_episodes,
                episode_grid=[final_episodes],
                update_freq=4,
                verbose_prefix=f"Exp10A-final/{cfg.name}/seed{seed}",
            )
            res = exp9.evaluate_policy(test_envs, lambda s, a=agent: a.select_action(s, training=False), n_runs=1)
            final_rows.append(_format_eval_row("final_test_by_seed", cfg, seed, res, best.get("best_episode")))
            final_seed_results.append(res)
            final_seed_states[f"{cfg.name}/seed{seed}"] = best.get("best_state")
        final_agg = exp9.aggregate_seed_results(final_seed_results)
        final_rows.append(_format_eval_row("final_test_summary", cfg, -1, final_agg, None))
        print(
            f"[Exp10-A-final] {cfg.name:18s} test_reward={final_agg['mean_reward']:.4f} "
            f"unmit={final_agg['unmitigated_rate']:.4f} events={final_agg['events_per_100_windows']:.4f}",
            flush=True,
        )

    final_df = pd.DataFrame(final_rows)
    _write_table(final_df, stage_dir / "exp10a_final_test.csv", stage_dir / "exp10a_final_test.md")
    metadata = {
        "stage": "Exp10-A",
        "script": Path(__file__).name,
        "fast": bool(args.fast),
        "seeds": seed_list,
        "episode_grid": episode_grid,
        "screen_episodes": screen_episodes,
        "final_episodes": final_episodes,
        "candidate_sources": _parse_sources(args.candidate_sources),
        "candidate_shard_index": args.candidate_shard_index,
        "candidate_shard_count": args.candidate_shard_count,
        "train_core": len(train_core),
        "validation": len(val_scenarios),
        "train_full": len(train_full),
        "test": len(test_scenarios),
        "active_fault_data_dir": str(exp9.get_active_fault_data_dir()),
        "feature_cols": exp9.get_feature_cols(exp9.COMMON.device),
        "diagnostic_report": diagnostic.report,
        "reward_protocol": "fixed Exp9-v6 CostConfig for every candidate; reward weights are not tuned.",
    }
    _write_json(stage_dir / "exp10a_metadata.json", metadata)
    return {
        "train_full": train_full,
        "test": test_scenarios,
        "raw_state_dim": raw_state_dim,
        "selected_configs": [asdict(c) for c in selected_cfgs],
        "best_overall": best_overall_name,
        "final_seed_states": final_seed_states,
        "metadata": metadata,
    }


def _build_policy_from_state(raw_state_dim: int, state: Optional[Any], seed: int = 11):
    cfg = _candidate_defaults(False)[0]
    agent = _make_qrdqn_agent(raw_state_dim, cfg, seed)
    if state is not None:
        agent.load_state(state)
    return agent


def _load_exp9_qrdqn_state(raw_state_dim: int) -> Tuple[Optional[Any], str]:
    paths = [
        OUTPUTS_DIR / "exp9_qrdqn_v6_model_comparison_parallel_merged" / "exp9_v6_models.pkl",
        OUTPUTS_DIR / "exp9_qrdqn_v6_model_comparison" / "exp9_v6_models.pkl",
    ]
    paths.extend(sorted(OUTPUTS_DIR.glob("exp9_qrdqn_v6_model_comparison*/exp9_v6_models.pkl")))
    for path in paths:
        if path.exists():
            import pickle

            with path.open("rb") as f:
                obj = pickle.load(f)
            states = obj.get("model_states", {}).get("QR-DQN", {})
            if states:
                first_key = sorted(states.keys())[0]
                return states[first_key], str(path)
    return None, "not_found"


def _counterfactual_action_scores(env: Any, horizon: int, followup_policy: Callable[[np.ndarray], int]) -> Dict[int, Dict[str, Any]]:
    scores: Dict[int, Dict[str, Any]] = {}
    for action in range(exp9.N_ACTIONS):
        trial = copy.deepcopy(env)
        state, reward, done, info = trial.step(action)
        total = float(reward)
        steps = 1
        unmitigated = int(bool(info.get("unmitigated", False)))
        low_risk_action = int(bool(info.get("low_zone", False) and info.get("action", 0) > 0))
        while not done and steps < horizon:
            next_action = int(followup_policy(state))
            state, reward, done, info = trial.step(next_action)
            total += float(reward) * (0.97 ** steps)
            unmitigated += int(bool(info.get("unmitigated", False)))
            low_risk_action += int(bool(info.get("low_zone", False) and info.get("action", 0) > 0))
            steps += 1
        scores[action] = {
            "return": total,
            "unmitigated_steps": unmitigated,
            "low_risk_actions": low_risk_action,
            "executed_first_action": int(info.get("action", action)),
        }
    return scores


def _supervisor_decision(state: np.ndarray, policy_action: int) -> Tuple[int, str]:
    risk = float(state[exp9.IDX_CURRENT_RISK] * 3.0)
    forecast = float(state[exp9.IDX_FORECAST_RISK] * 3.0)
    diag_conf = float(state[exp9.IDX_DIAG_CONF])
    entropy = float(state[exp9.IDX_DIAG_ENTROPY])
    if policy_action == 0 and max(risk, forecast) >= 1.20:
        return 3, "escalate_continue_in_high_risk"
    if policy_action == 0 and max(risk, forecast) >= 0.80 and diag_conf >= 0.45:
        return 2, "escalate_continue_in_warning_zone"
    if policy_action >= 2 and max(risk, forecast) < 0.45 and diag_conf < 0.30:
        return 1, "downgrade_low_risk_maintenance_to_inspection"
    if entropy >= 0.72 and 0.55 <= max(risk, forecast) <= 1.05:
        return 1, "request_inspection_under_diagnostic_uncertainty"
    return int(policy_action), "approve_policy_action"


def _state_evidence(state: np.ndarray) -> Dict[str, Any]:
    probs = state[exp9.IDX_DIAG_START:exp9.IDX_DIAG_END]
    return {
        "risk": round(float(state[exp9.IDX_CURRENT_RISK] * 3.0), 4),
        "forecast_risk": round(float(state[exp9.IDX_FORECAST_RISK] * 3.0), 4),
        "trend": round(float(state[exp9.IDX_TREND]), 4),
        "volatility": round(float(state[exp9.IDX_VOLATILITY]), 4),
        "diag_probabilities": {
            ft: round(float(probs[i]), 4)
            for i, ft in enumerate(exp9.FAULT_TYPES)
        },
        "diag_conf": round(float(state[exp9.IDX_DIAG_CONF]), 4),
        "diag_entropy": round(float(state[exp9.IDX_DIAG_ENTROPY]), 4),
        "time_since_action": round(float(state[exp9.IDX_TIME_SINCE_ACTION] * 30.0), 4),
        "last_action_vector": [round(float(x), 4) for x in state[exp9.IDX_LAST_ACTION_START:exp9.IDX_LAST_ACTION_START + exp9.N_ACTIONS]],
    }


def run_exp10b(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    stage_dir = out_dir / "exp10b_decision_audit"
    stage_dir.mkdir(parents=True, exist_ok=True)
    train_full, test_scenarios, diagnostic = _load_splits(args.fast, out_dir)
    train_envs = _make_envs(train_full, seed=4500)
    test_envs = _make_envs(test_scenarios, seed=5000)
    raw_state_dim = test_envs[0].state_dim
    state_obj, state_source = _load_exp9_qrdqn_state(raw_state_dim)
    agent = _build_policy_from_state(raw_state_dim, state_obj, seed=11)
    policy = lambda s: agent.select_action(s, training=False)
    if state_obj is None:
        train_episodes = int(args.audit_train_episodes if args.audit_train_episodes is not None else (5 if args.fast else 300))
        print(f"[WARN] No saved Exp9 QR-DQN state found; training an audit QR-DQN for {train_episodes} episodes.", flush=True)
        _, _, best = exp9.train_agent(
            agent,
            train_envs,
            test_envs,
            n_episodes=train_episodes,
            episode_grid=[train_episodes],
            update_freq=4,
            verbose_prefix="Exp10B-audit-policy",
        )
        state_source = f"trained_in_exp10b_best_episode_{best.get('best_episode')}"

    rng = np.random.default_rng(args.random_seed)
    env_indices = np.arange(len(test_envs))
    rng.shuffle(env_indices)
    case_rows: List[Dict[str, Any]] = []
    packet_rows: List[Dict[str, Any]] = []
    gold_rows: List[Dict[str, Any]] = []
    max_cases = int(args.audit_cases)
    per_env_limit = max(1, math.ceil(max_cases / max(1, min(len(test_envs), 16))))

    for env_idx in env_indices:
        if len(case_rows) >= max_cases:
            break
        env = test_envs[int(env_idx)]
        state = env.reset()
        done = False
        picked = 0
        while not done and picked < per_env_limit and len(case_rows) < max_cases:
            policy_action = int(policy(state))
            risk = float(state[exp9.IDX_CURRENT_RISK] * 3.0)
            forecast = float(state[exp9.IDX_FORECAST_RISK] * 3.0)
            diag_conf = float(state[exp9.IDX_DIAG_CONF])
            interesting = (
                max(risk, forecast) >= 0.65
                or diag_conf >= 0.45
                or policy_action > 0
                or rng.random() < (0.05 if not args.fast else 0.20)
            )
            if interesting:
                followup = lambda ss: agent.select_action(ss, training=False)
                cf = _counterfactual_action_scores(env, horizon=args.audit_horizon, followup_policy=followup)
                oracle_action = max(cf, key=lambda a: cf[a]["return"])
                supervisor_action, supervisor_reason = _supervisor_decision(state, policy_action)
                case_id = f"audit_{len(case_rows)+1:04d}"
                public_packet = {
                    "case_id": case_id,
                    "state_evidence": _state_evidence(state),
                    "policy_action": exp9.ACTION_NAMES[policy_action],
                    "agent_supervisor_action": exp9.ACTION_NAMES[supervisor_action],
                    "agent_supervisor_reason": supervisor_reason,
                    "allowed_actions": exp9.ACTION_NAMES,
                    "evidence_note": "Online evidence only; ground-truth fault labels and counterfactual oracle are private.",
                }
                gold = {
                    "case_id": case_id,
                    "scenario_private": env.scenario.name,
                    "fault_type_private": env.scenario.fault_type,
                    "severity_private": env.scenario.severity_level,
                    "percentage_private": env.scenario.percentage,
                    "step": int(env.t),
                    "policy_action_id": policy_action,
                    "supervisor_action_id": supervisor_action,
                    "oracle_action_id": int(oracle_action),
                    "counterfactual_scores": cf,
                }
                packet_rows.append(public_packet)
                gold_rows.append(gold)
                case_rows.append({
                    "case_id": case_id,
                    "policy_action": exp9.ACTION_NAMES[policy_action],
                    "supervisor_action": exp9.ACTION_NAMES[supervisor_action],
                    "oracle_action": exp9.ACTION_NAMES[int(oracle_action)],
                    "policy_oracle_match": int(policy_action == oracle_action),
                    "supervisor_oracle_match": int(supervisor_action == oracle_action),
                    "policy_regret": round(float(cf[int(oracle_action)]["return"] - cf[policy_action]["return"]), 5),
                    "supervisor_regret": round(float(cf[int(oracle_action)]["return"] - cf[supervisor_action]["return"]), 5),
                    "risk": round(risk, 5),
                    "forecast_risk": round(forecast, 5),
                    "diag_conf": round(diag_conf, 5),
                    "supervisor_reason": supervisor_reason,
                })
                picked += 1
            state, _, done, _ = env.step(policy_action)

    case_df = pd.DataFrame(case_rows)
    _write_table(case_df, stage_dir / "exp10b_action_audit_cases.csv", stage_dir / "exp10b_action_audit_cases.md")
    _write_jsonl(stage_dir / "exp10b_public_evidence_packets.jsonl", packet_rows)
    _write_jsonl(stage_dir / "exp10b_private_gold.jsonl", gold_rows)
    summary = {
        "stage": "Exp10-B",
        "cases": len(case_rows),
        "policy_oracle_agreement": float(case_df["policy_oracle_match"].mean()) if not case_df.empty else None,
        "supervisor_oracle_agreement": float(case_df["supervisor_oracle_match"].mean()) if not case_df.empty else None,
        "policy_mean_regret": float(case_df["policy_regret"].mean()) if not case_df.empty else None,
        "supervisor_mean_regret": float(case_df["supervisor_regret"].mean()) if not case_df.empty else None,
        "qrdqn_state_source": state_source,
        "diagnostic_report": diagnostic.report,
    }
    _write_json(stage_dir / "exp10b_summary.json", summary)
    print(f"[Exp10-B] cases={summary['cases']} policy_agree={summary['policy_oracle_agreement']} supervisor_agree={summary['supervisor_oracle_agreement']}", flush=True)
    return summary


def _build_exp10c_packets(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    b_dir = out_dir / "exp10b_decision_audit"
    packets_path = b_dir / "exp10b_public_evidence_packets.jsonl"
    gold_path = b_dir / "exp10b_private_gold.jsonl"
    if not packets_path.exists() or not gold_path.exists():
        run_exp10b(args, out_dir)
    audit_packets = _read_jsonl(packets_path)
    gold_rows = _read_jsonl(gold_path)
    gold_by_case = {g["case_id"]: g for g in gold_rows}
    selected = audit_packets[: int(args.qa_cases)]
    qa_packets: List[Dict[str, Any]] = []
    qa_gold: List[Dict[str, Any]] = []
    for packet in selected:
        case_id = packet["case_id"]
        qa_packets.append({
            "case_id": case_id,
            "question": (
                "Based on the online predictive-maintenance evidence, explain the equipment condition, "
                "whether the QR-DQN action should be accepted, and what action should be recommended."
            ),
            "evidence": packet,
            "required_output": {
                "condition_assessment": "concise equipment condition assessment",
                "action_audit": "approve/escalate/downgrade/request inspection with reason",
                "recommended_action": "one of Continue, Inspect, Preventive, Immediate",
                "evidence_used": "list concrete evidence values used",
                "uncertainty": "state the main uncertainty",
            },
            "leakage_control": "No ground truth, scenario name, severity, percentage, or oracle label is included.",
        })
        private = gold_by_case[case_id]
        qa_gold.append({
            "case_id": case_id,
            "fault_type_private": private["fault_type_private"],
            "oracle_action_private": exp9.ACTION_NAMES[int(private["oracle_action_id"])],
            "policy_action_private": exp9.ACTION_NAMES[int(private["policy_action_id"])],
            "supervisor_action_private": exp9.ACTION_NAMES[int(private["supervisor_action_id"])],
        })
    return qa_packets, qa_gold


def _qa_prompt(packet: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "instruction": (
                "You are an industrial predictive-maintenance expert. Use only the supplied online evidence. "
                "Do not invent ground truth, dates, hidden labels, or sensor values. Return concise JSON."
            ),
            **packet,
        },
        ensure_ascii=False,
        indent=2,
    )


def _parse_models(args: argparse.Namespace) -> List[LLMModel]:
    if args.models:
        out = []
        for item in args.models.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                display, model_id = item.split("=", 1)
                out.append(LLMModel(display.strip(), model_id.strip()))
            else:
                out.append(LLMModel(item, item))
        return out
    return MODEL_PRESETS[args.model_preset]


def _infer_action_from_text(text: str) -> Optional[str]:
    lower = text.lower()
    mapping = {
        "Immediate": ["immediate", "replace", "shutdown", "urgent"],
        "Preventive": ["preventive", "planned maintenance", "schedule maintenance"],
        "Inspect": ["inspect", "inspection", "check"],
        "Continue": ["continue", "monitor"],
    }
    for action, keywords in mapping.items():
        if any(k in lower for k in keywords):
            return action
    return None


def run_exp10c(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    stage_dir = out_dir / "exp10c_leakfree_qa"
    stage_dir.mkdir(parents=True, exist_ok=True)
    packets, gold = _build_exp10c_packets(args, out_dir)
    models = _parse_models(args)
    _write_jsonl(stage_dir / "exp10c_public_evidence_packets.jsonl", packets)
    _write_jsonl(stage_dir / "exp10c_private_gold.jsonl", gold)

    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SF_API_KEY")
    records: List[Dict[str, Any]] = []
    for packet in packets:
        prompt = _qa_prompt(packet)
        for model in models:
            row = {
                "case_id": packet["case_id"],
                "display_name": model.display_name,
                "model_id": model.model_id,
                "prompt": prompt,
                "evidence_packet": packet,
                "ok": True,
                "latency_sec": 0.0,
                "answer_text": "",
                "raw_response": None,
                "error": None,
            }
            if args.dry_run_llm or not api_key:
                row["answer_text"] = (
                    "Dry-run response: evidence reviewed. Recommended action follows the provided supervisor action."
                )
                if not args.dry_run_llm and not api_key:
                    row["ok"] = False
                    row["error"] = "missing SILICONFLOW_API_KEY/SF_API_KEY"
            else:
                try:
                    content, raw, latency = _call_chat_completion(
                        model_id=model.model_id,
                        prompt=prompt,
                        api_key=api_key,
                        endpoint=args.llm_endpoint,
                        temperature=args.qa_temperature,
                        max_tokens=args.qa_max_tokens,
                        timeout=args.llm_timeout,
                    )
                    row.update({"answer_text": content, "raw_response": raw, "latency_sec": latency})
                except Exception as exc:
                    row.update({"ok": False, "error": repr(exc)})
            records.append(row)
            status = "OK" if row["ok"] else "FAIL"
            print(f"[Exp10-C] {status} {packet['case_id']} {model.display_name} {row.get('latency_sec')}", flush=True)

    gold_by_case = {g["case_id"]: g for g in gold}
    metric_rows = []
    for rec in records:
        gold_action = gold_by_case[rec["case_id"]]["oracle_action_private"]
        pred_action = _infer_action_from_text(str(rec.get("answer_text", "")))
        metric_rows.append({
            "case_id": rec["case_id"],
            "display_name": rec["display_name"],
            "ok": int(bool(rec.get("ok"))),
            "latency_sec": rec.get("latency_sec"),
            "oracle_action": gold_action,
            "parsed_recommended_action": pred_action or "UNKNOWN",
            "action_match": int(pred_action == gold_action) if pred_action else 0,
        })
    records_path = stage_dir / "exp10c_qa_records.jsonl"
    _write_jsonl(records_path, records)
    metric_df = pd.DataFrame(metric_rows)
    _write_table(metric_df, stage_dir / "exp10c_objective_metrics_by_case.csv")
    summary_df = metric_df.groupby("display_name", as_index=False).agg(
        Success_rate=("ok", "mean"),
        Action_agreement=("action_match", "mean"),
        Mean_latency_sec=("latency_sec", "mean"),
        Cases=("case_id", "count"),
    )
    _write_table(summary_df, stage_dir / "exp10c_model_summary.csv", stage_dir / "exp10c_model_summary.md")
    summary = {
        "stage": "Exp10-C",
        "records": len(records),
        "cases": len(packets),
        "models": [asdict(m) for m in models],
        "dry_run_llm": bool(args.dry_run_llm),
        "leakage_control": "Public prompts exclude ground truth, scenario names, severity, percentage, and oracle labels.",
    }
    _write_json(stage_dir / "exp10c_summary.json", summary)
    return summary


def run_all(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir).resolve() if args.out_dir else OUTPUTS_DIR / f"exp10_v3_formal_agent_loop_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "exp10_v3_run_config.json", {
        "script": Path(__file__).name,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "active_fault_data_dir": str(exp9.get_active_fault_data_dir()),
        "feature_cols": exp9.get_feature_cols(exp9.COMMON.device),
    })

    stages = _parse_sources(args.stages)
    results: Dict[str, Any] = {}
    if "a" in stages:
        results["a"] = run_exp10a(args, out_dir)
    if "b" in stages:
        results["b"] = run_exp10b(args, out_dir)
    if "c" in stages:
        results["c"] = run_exp10c(args, out_dir)
    _write_json(out_dir / "exp10_v3_completion_summary.json", {
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "stages": stages,
        "result_keys": list(results),
        "output_dir": str(out_dir),
    })
    print(f"[Exp10-v3] outputs written to: {out_dir}", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp10 v3 formal agent-in-the-loop experiment.")
    parser.add_argument("--stages", default="a,b,c", help="Comma-separated stages: a,b,c.")
    parser.add_argument("--fast", action="store_true", help="Use fast scenario subset.")
    parser.add_argument("--out-dir", default=None, help="Output directory.")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--episode-grid", default="100,300,500")
    parser.add_argument("--screen-episodes", type=int, default=None)
    parser.add_argument("--final-episodes", type=int, default=None)
    parser.add_argument("--candidate-sources", default="default,random,agent")
    parser.add_argument("--candidates-per-source", type=int, default=3)
    parser.add_argument("--candidate-shard-index", type=int, default=0)
    parser.add_argument("--candidate-shard-count", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=20260711)
    parser.add_argument("--audit-cases", type=int, default=96)
    parser.add_argument("--audit-horizon", type=int, default=16)
    parser.add_argument("--audit-train-episodes", type=int, default=None)
    parser.add_argument("--qa-cases", type=int, default=24)
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="core")
    parser.add_argument("--models", default=None, help="Display=model_id comma list; overrides preset.")
    parser.add_argument("--agent-model", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--llm-endpoint", default=DEFAULT_SF_ENDPOINT)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--qa-temperature", type=float, default=0.2)
    parser.add_argument("--qa-max-tokens", type=int, default=1200)
    parser.add_argument("--dry-run-llm", action="store_true", help="Do not call external LLM APIs.")
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
