"""Frozen-checkpoint Exp3 fusion study. Select on development dates, then evaluate."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

import evaluate_exp3_revision_alarm as replay
from evaluate_exp3_revision_causal_fill import dense_input


CORE = ("latent", "reconstruction", "residual")
EXTRA = ("vibration_level", "vibration_change", "warming")
NAMES = (*CORE, *EXTRA, "locked")
METRICS = ("auc_roc", "precision", "recall", "fpr")


def case_plan(spec, stage):
    keys = ("tune", "check") if stage == "dev" else ("test",)
    for key in keys:
        for day in spec[key]:
            for mechanism in replay.FAULTS:
                for level in ((4, 6, 8) if stage == "dev" else range(1, 11)):
                    for pct in ((5,) if stage == "dev" else replay.FRACTIONS):
                        yield key, day, mechanism, level, pct, f"{day}_{mechanism}_pct{pct}_sev{level}"


def load_background(path):
    frame = pd.read_csv(path, usecols=["Timestamp", *replay.FEATURES])
    return pd.to_datetime(frame.Timestamp).to_numpy(dtype="datetime64[s]"), frame[replay.FEATURES].to_numpy(
        dtype=np.float32, copy=True)


def prepare(args):
    spec = json.loads((args.base / "seed_42_causal_fill/protocol.json").read_text(encoding="utf-8"))
    dev = sorted(spec["development"])
    assert len(dev) == 24 and len(spec["test"]) == 5 and len(spec["train"]) == 29
    assert not (set(dev) & set(spec["test"]))
    plan = dict(calibration=dev[::3], tune=dev[1::3], check=dev[2::3],
                test=spec["test"], train=spec["train"], hashes={}, fits={}, masks={},
                config=spec["config_sha256"], locked=replay.sha(replay.LOCK / "exp3_model_selection.py"),
                driver=replay.sha(Path(__file__)), replay=replay.sha(Path(replay.__file__)),
                protocol_source=replay.sha(args.base / "seed_42_causal_fill/protocol.json"))
    args.out.mkdir(parents=True, exist_ok=False)
    for seed in replay.SEEDS:
        base = args.base / ("seed_42" if seed == 42 else f"seed_{seed}_fit")
        path = args.base / f"seed_{seed}_causal_fill"
        manifest = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
        assert replay.sha(base / "model.pt") == manifest["checkpoint_sha256"]
        assert manifest["development"] == spec["development"] and manifest["test"] == spec["test"]
        plan["fits"][str(seed)] = dict(model=replay.sha(base / "model.pt"),
                                      reference=replay.sha(base / "reference.npz"))
    paths = []
    for day in [*dev, *plan["test"]]:
        ts, raw, state = replay.read_day(day)
        prep = replay.DAILY / f"{day}_preprocessed.csv"
        base_t, preprocessed = load_background(prep)
        observed = np.isfinite(raw)
        np.testing.assert_array_equal(ts, base_t)
        np.testing.assert_array_equal(raw[observed], preprocessed[observed])
        mask = args.out / f"{day}.npz"
        np.savez_compressed(mask, observed=observed, timestamp=ts, state=state)
        plan["masks"][day] = replay.sha(mask)
        paths.extend([prep, replay.LABELS / f"{day}_labels.csv"])
    for stage in ("dev", "test"):
        for *_, name in case_plan(plan, stage):
            paths.extend(replay.SCENARIOS / f"{name}_{part}.parquet" for part in ("data", "labels"))
    plan["hashes"] = {p.relative_to(replay.ROOT).as_posix(): replay.sha(p) for p in paths}
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Frozen split: {len(dev)} development, {len(plan['test'])} replay dates; {len(paths)} input hashes.")


def read_plan(args):
    plan = json.loads((args.plan / "plan.json").read_text(encoding="utf-8"))
    assert plan["driver"] == replay.sha(Path(__file__))
    assert plan["replay"] == replay.sha(Path(replay.__file__))
    return plan


def configure(root, lock):
    replay.ROOT = root
    replay.DAILY = root / "data/monitoring/line_1"
    replay.LABELS = root / "data/labels/line_1"
    replay.SCENARIOS = root / "data/scenarios/line_1"
    replay.LOCK = lock


def verify(args):
    plan = read_plan(args)
    for file, digest in plan["hashes"].items():
        if replay.sha(args.root / file) != digest:
            raise ValueError(f"Input differs from local archive: {file}")
    for day, digest in plan["masks"].items():
        assert replay.sha(args.plan / f"{day}.npz") == digest
    args.out.write_text(replay.sha(args.plan / "plan.json"), encoding="utf-8")
    print(f"Verified {len(plan['hashes'])} server source hashes and all observation masks.")


def extra_features(scaled):
    # Compare the current seven points with older observations, leaving a 30 s gap.
    c = np.vstack([np.zeros((1, 4)), scaled.cumsum(axis=0, dtype=np.float64)])
    sq = np.vstack([np.zeros((1, 4)), np.square(scaled).cumsum(axis=0, dtype=np.float64)])
    t = np.arange(len(scaled))
    left, right = np.maximum(t-3, 0), np.minimum(t+4, len(scaled))
    recent = (c[right] - c[left]) / (right-left)[:, None]
    older_start, older_stop = np.maximum(t-330, 0), np.maximum(t-30, 0)
    n = np.maximum(older_stop-older_start, 1)[:, None]
    past = (c[older_stop]-c[older_start]) / n
    var = (sq[older_stop]-sq[older_start]) / n - past**2
    contrast = (recent-past) / np.sqrt(np.maximum(var, .35**2))
    contrast[n[:, 0] < 30] = 0
    return np.column_stack([np.max(np.abs(recent[:, :3]), axis=1),
                            np.max(np.abs(contrast[:, :3]), axis=1),
                            np.maximum(contrast[:, 3], 0)])


def component_scores(mod, model, scaler, ref, ts, values):
    index, dense_ts, filled = dense_input(ts, values)
    output = np.full((len(dense_ts), len(NAMES)), np.nan, dtype=np.float32)
    for start, stop in replay.spans(dense_ts, np.isfinite(filled).all(axis=1)):
        scaled = scaler.transform(filled[start:stop]).astype(np.float32)
        z, e, r = replay.infer(mod, model, scaled)
        a = mod._dist_to_baseline(z, ref["latent"], "DBSCAN")
        b = mod._dist_to_baseline(r, ref["residual"], "DBSCAN")
        stats = ref["stats"]
        standardized = np.column_stack([
            (x-stats[2*i])/(stats[2*i+1]+1e-12) for i,x in enumerate((a,e,b))
        ])
        locked = mod._score_ours_ab(a, e, b, *stats)
        all_scores = np.column_stack([standardized, extra_features(scaled), locked])
        output[start+replay.DELAY:stop-replay.DELAY] = all_scores[replay.DELAY:-replay.DELAY]
    return output[index]


def extract(args):
    plan = read_plan(args)
    assert args.receipt.read_text(encoding="utf-8") == replay.sha(args.plan / "plan.json")
    configure(args.root, args.lock)
    assert plan["config"] == replay.sha(replay.LOCK / "config.py")
    if args.stage == "test":
        frozen = json.loads(args.selection.read_text(encoding="utf-8"))
        assert frozen["plan_sha"] == replay.sha(args.plan / "plan.json")
    sys.path.insert(0, str(replay.LOCK))
    mod, _ = replay.load_locked()
    torch.set_num_threads(3)
    mod._set_device(0)
    fit = plan["fits"][str(args.seed)]
    assert fit["model"] == replay.sha(args.fit / "model.pt")
    assert fit["reference"] == replay.sha(args.fit / "reference.npz")
    with np.load(args.fit / "reference.npz") as source:
        ref = dict(source)
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = ref["mean"], ref["scale"]
    model = mod._build_ae("Conv1DAE", 4, 8, [], []).to(mod.DEVICE)
    model.load_state_dict(torch.load(args.fit / "model.pt", map_location=mod.DEVICE, weights_only=True))
    args.out.mkdir(parents=True, exist_ok=False)
    days = {}
    roles = ("calibration", "tune", "check") if args.stage == "dev" else ("test",)
    for role in roles:
        for day in plan[role]:
            ts, raw = load_background(replay.DAILY / f"{day}_preprocessed.csv")
            with np.load(args.plan / f"{day}.npz") as mask:
                np.testing.assert_array_equal(ts, mask["timestamp"])
                observed, state = mask["observed"], mask["state"]
            days[day] = ts, observed, state
            if role == "calibration":
                raw[~observed] = np.nan
                s = component_scores(mod, model, scaler, ref, ts, raw)
                valid = np.isfinite(s).all(axis=1) & (state == 2)
                np.savez_compressed(args.out / f"cal_{day}.npz", x=s[valid])
    rows = []
    for role, day, mechanism, level, pct, name in case_plan(plan, args.stage):
        ts, observed, state = days[day]
        values, fault = replay.read_scenario(name, ts, state)
        values = np.where(observed, values, np.nan)
        s = component_scores(mod, model, scaler, ref, ts, values)
        keep = (fault | (state == 2)) & np.isfinite(s).all(axis=1)
        np.savez_compressed(args.out / f"{name}.npz", x=s[keep], y=fault[keep])
        rows.append(dict(role=role, day=day, mechanism=mechanism, level=level, pct=pct,
                         scenario=name, eligible=int(keep.sum()), positive=int((keep & fault).sum()),
                         fault_total=int(fault.sum())))
        if len(rows) % 48 == 0:
            print(f"Seed {args.seed}, {args.stage}: {len(rows)} scenarios", flush=True)
    expected = 192 if args.stage == "dev" else 600
    assert len(rows) == expected
    pd.DataFrame(rows).to_csv(args.out / "cases.csv", index=False)
    (args.out / "verified.json").write_text(json.dumps(dict(seed=args.seed, stage=args.stage,
        count=len(rows), plan_sha=replay.sha(args.plan / "plan.json"), device=str(mod.DEVICE)),
        indent=2), encoding="utf-8")


def candidate_bank():
    bank = {"locked": {"mode":"locked"}}
    # Coarse simplex on the existing components and one fixed three-component max rule.
    for i in range(5):
        for j in range(5-i):
            weights = [i/4, j/4, (4-i-j)/4, 0, 0, 0]
            for mode in ("linear", "tail"):
                bank[f"{mode}_{i}{j}{4-i-j}"] = dict(mode=mode, weights=weights)
    bank["tail_max_core"] = dict(mode="max", dims=[0,1,2])
    bank["tail_max_all"] = dict(mode="max", dims=list(range(6)))
    bank["tail_max_local"] = dict(mode="max", dims=[3,4,5])
    bank["tail_max_latent_local"] = dict(mode="max", dims=[0,3,4,5])
    for extra in range(3,6):
        for weight in (.25,.5,.75,1.):
            weights = [(1-weight)*a for a in (.5,.25,.25,0,0,0)]
            weights[extra] = weight
            bank[f"tail_extra_{extra}_{weight}"] = dict(mode="tail", weights=weights)
    bank["tail_equal_all"] = dict(mode="tail", weights=[1/6]*6)
    return bank


def transform(x, cal):
    return np.column_stack([
        np.log((len(cal)+1) / (1+len(cal)-np.searchsorted(cal[:,j], x[:,j], side="left")))
        for j in range(6)
    ]).astype(np.float32)


def score_bank(x, cal, bank):
    scores = np.empty((len(x),len(bank)),dtype=np.float32)
    tail = transform(x, cal)
    for i,spec in enumerate(bank.values()):
        mode = spec["mode"]
        if mode == "locked":
            scores[:,i] = x[:,-1]
        elif mode == "max":
            scores[:,i] = tail[:,spec["dims"]].max(axis=1)
        else:
            arr = x[:,:6] if mode=="linear" else tail
            scores[:,i] = arr @ np.asarray(spec["weights"], dtype=np.float32)
    return scores


def read_batch(folder, table):
    parts = []
    for scenario in table.scenario:
        with np.load(folder / f"{scenario}.npz") as a:
            parts.append((a["x"],a["y"]))
    return tuple(np.concatenate(v) for v in zip(*parts))


def evaluate_bank(root, plan_sha, bank, stage, use_cal, cutoffs=None, auc=False):
    rows = []
    limits = {}
    for seed in replay.SEEDS:
        folder = root / f"{stage}_{seed}"
        manifest = json.loads((folder / "verified.json").read_text(encoding="utf-8"))
        assert manifest["plan_sha"] == plan_sha and manifest["seed"] == seed
        data = pd.read_csv(folder / "cases.csv")
        cal = []
        for file in sorted((use_cal / f"dev_{seed}").glob("cal_*.npz")):
            with np.load(file) as f:
                cal.append(f["x"])
        assert len(cal) == 8
        cal = np.concatenate(cal)
        ordered = np.sort(cal[:,:6],axis=0)
        if cutoffs is None:
            calibration = score_bank(cal, ordered, bank)
            threshold = np.quantile(calibration, .99, axis=0, method="higher")
            assert ((calibration>threshold).mean(0)<=.01).all()
        else:
            threshold = np.asarray([cutoffs[str(seed)][k] for k in bank], dtype=np.float32)
        limits[str(seed)] = dict(zip(bank, map(float,threshold)))
        for (role,fault,level),table in data.groupby(["role","mechanism","level"]):
            x,y = read_batch(folder,table)
            assert len(y) == table.eligible.sum() and int(y.sum()) == table.positive.sum()
            scores = score_bank(x, ordered, bank)
            hits = scores>threshold
            tp = hits[y].sum(0); fp = hits[~y].sum(0)
            for j,name in enumerate(bank):
                rows.append(dict(seed=seed,role=role,mechanism=fault,level=level,candidate=name,
                    tp=int(tp[j]),fp=int(fp[j]),fn=int(y.sum()-tp[j]),tn=int((~y).sum()-fp[j]),
                    precision=tp[j]/(tp[j]+fp[j]) if tp[j]+fp[j] else np.nan,
                    recall=tp[j]/y.sum(),fpr=fp[j]/(~y).sum(),
                    auc_roc=roc_auc_score(y,scores[:,j]) if auc else np.nan,
                    coverage=int(y.sum())/table.fault_total.sum()))
        print(f"Evaluated {stage} seed {seed} with {len(bank)} candidate scores.",flush=True)
    return pd.DataFrame(rows),limits


def select(args):
    plan = read_plan(args)
    plan_sha = replay.sha(args.plan / "plan.json")
    bank = candidate_bank()
    frame, limits = evaluate_bank(args.runs, plan_sha, bank, "dev", args.runs)
    train = frame.loc[frame.role.eq("tune")]
    checks = frame.loc[frame.role.eq("check")]
    recall = train.groupby(["candidate","mechanism"]).recall.mean().unstack()
    rank = recall.assign(min_recall=recall.min(axis=1), macro_recall=recall.mean(axis=1),
                         fpr=train.groupby("candidate").fpr.mean())
    rank["qualifies"] = rank.fpr.le(max(.015, rank.loc["locked","fpr"]+.002))
    rank = rank.sort_values(["qualifies","min_recall","macro_recall","fpr"],
                            ascending=[False,False,False,True])
    best = rank.index[0]
    # Report best linear separately to expose any benefit from the new feature family.
    linear = next((n for n in rank.index if n.startswith("linear_") and rank.loc[n,"qualifies"]), "locked")
    chosen = dict.fromkeys(["locked",best,linear])
    picked = {name:bank[name] for name in chosen}
    validation = checks.loc[checks.candidate.isin(picked)].groupby(["candidate","mechanism"])[
        ["precision","recall","fpr"]].mean().reset_index()
    args.out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out / "development.csv",index=False)
    rank.to_csv(args.out / "selection_rank.csv")
    validation.to_csv(args.out / "development_check.csv",index=False)
    decision = dict(plan_sha=plan_sha, candidates=picked, selected=best, best_linear=linear,
        cutoffs={s:{name:limits[s][name] for name in picked} for s in limits},
        gate="Tuning macro FPR <= max(1.5%, recalibrated locked FPR + 0.20 pp)",
        objective="maximize worst-fault mean recall at levels 4,6,8, then macro recall; no test inputs used")
    (args.out / "selection.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
    print(rank.head(8).to_string(),flush=True)
    print(validation.to_string(index=False),flush=True)


def test_report(args):
    frozen = json.loads(args.selection.read_text(encoding="utf-8"))
    plan_sha = replay.sha(args.plan / "plan.json")
    assert frozen["plan_sha"] == plan_sha
    frame,_ = evaluate_bank(args.runs,plan_sha,frozen["candidates"],"test",args.runs,
                            cutoffs=frozen["cutoffs"],auc=True)
    assert len(frame) == len(frozen["candidates"])*200
    args.out.mkdir(parents=True,exist_ok=False)
    frame.to_csv(args.out / "by_seed.csv",index=False)
    for name,part in frame.groupby("candidate"):
        stats=part.groupby(["mechanism","level"])[[*METRICS,"coverage"]].agg(["mean","std","count"])
        stats.columns=[f"{k}_{tag}" for k,tag in stats.columns]
        stats.reset_index().to_csv(args.out / f"{name}_by_fault_and_level.csv",index=False)
    mids = frame.loc[frame.level.between(4,8)].groupby(["candidate","mechanism"])[list(METRICS)].mean()
    mids.to_csv(args.out / "levels_4_to_8.csv")
    print(mids.to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(required=True)
    for name,action,fields in (
        ("prepare",prepare,("base","out")),
        ("verify",verify,("plan","root","out")),
        ("extract",extract,("plan","root","lock","fit","receipt","out")),
        ("select",select,("plan","runs","out")),
        ("test-report",test_report,("plan","runs","selection","out")),
    ):
        p=commands.add_parser(name)
        for field in fields:
            p.add_argument(f"--{field}",type=Path,required=True)
        if name=="extract":
            p.add_argument("--seed",type=int,choices=replay.SEEDS,required=True)
            p.add_argument("--stage",choices=("dev","test"),required=True)
            p.add_argument("--selection",type=Path)
        p.set_defaults(action=action)
    args=parser.parse_args()
    args.action(args)


if __name__=="__main__":
    main()
