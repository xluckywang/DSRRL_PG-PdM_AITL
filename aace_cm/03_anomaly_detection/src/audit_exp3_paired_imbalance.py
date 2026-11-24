"""Compare the archived imbalance scores with same-time, uninjected monitoring windows."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

import evaluate_exp3_revision_alarm as replay
import exp3_context_revision as ctx
import exp3_fusion_revision as fusion


LEVELS = range(4, 11)
PARTS = ["latent", "reconstruction", "residual", "warming"]
COLS = [0, 1, 2, 5]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_scorer(x, z, incumbent, candidate):
    normal = np.sort(x[:, :6], axis=0)
    combined = np.column_stack([x[:, COLS], fusion.score_bank(x, normal, incumbent)[:, 0]])
    raw_ref = np.sort(combined, axis=0)
    option = next(iter(candidate.values()))
    weights = np.array(next(iter(incumbent.values()))["weights"])[COLS]
    dims = tuple(option[0]) if option is not None else ()
    models = {dims: ctx.fit_reference(z, combined, dims)} if dims else {}

    def score(x, z):
        final = ctx.scored(x, z, normal, incumbent, models, raw_ref, weights, candidate)[:, 0]
        parts = (fusion.transform(x, normal)[:, COLS] * weights).astype(np.float32)
        return np.column_stack([final, parts])

    return score


def export(root, out):
    ph = replay.sha(root / "plan/plan.json")
    prior_file = root / "frozen/selection.json"
    prior, selection = read_json(prior_file), read_json(root / "context_frozen/selection.json")
    assert prior["plan_sha"] == selection["plan_sha"] == ph
    assert selection["incumbent_sha"] == replay.sha(prior_file)
    name = selection["selected"]
    inc = {prior["selected"]: prior["candidates"][prior["selected"]]}
    assert next(iter(inc.values()))["mode"] == "tail"
    candidate = {name: selection["options"][name]}
    prior_results = pd.read_csv(root / "context_results/by_seed.csv")
    prior_results = prior_results[prior_results.candidate.eq(name) & prior_results.mechanism.eq("imbalance")]
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(dict(plan_sha=ph, incumbent=inc, candidate=candidate,
        cutoffs=selection["cutoffs"]), indent=2), encoding="utf-8")
    for seed in replay.SEEDS:
        dev, test = root / f"dev_{seed}", root / f"test_{seed}"
        calctx, testctx = root / "context_dev", root / "context_test"
        for p in (dev, test, calctx, testctx):
            assert read_json(p / "verified.json")["plan_sha"] == ph
        files = sorted(dev.glob("cal_*.npz"))
        assert len(files) == 8
        xp, zp = [], []
        for f in files:
            x, = ctx.read(f)
            z, = ctx.read(calctx / f.name)
            assert len(x) == len(z)
            xp.append(x)
            zp.append(z)
        x, z = np.concatenate(xp), np.concatenate(zp)
        score = build_scorer(x, z, inc, candidate)
        cutoff = np.float32(selection["cutoffs"][str(seed)][name])
        assert np.quantile(score(x,z)[:,0], .99, method="higher") == cutoff
        directory = out / str(seed)
        directory.mkdir()
        np.savez_compressed(directory / "cal.npz", x=x, z=z)
        cases = pd.read_csv(test / "cases.csv")
        cases = cases[cases.mechanism.eq("imbalance") & cases.level.isin(LEVELS)]
        assert len(cases) == 105 and cases.role.eq("test").all()
        rows = []
        for c in cases.itertuples(index=False):
            x, y = ctx.read(test / f"{c.scenario}.npz")
            z, y2 = ctx.read(testctx / f"{c.scenario}.npz")
            np.testing.assert_array_equal(y, y2)
            assert len(y) == c.eligible and y.sum() == c.positive
            vals = score(x, z)
            hit = vals[:,0] > cutoff
            rows.append(dict(level=c.level, tp=(hit & y).sum(), fp=(hit & ~y).sum(),
                             fn=(~hit & y).sum(), tn=(~hit & ~y).sum()))
            np.savez_compressed(directory / f"{c.scenario}.npz", y=y, positive=vals[y])
        archived = prior_results[prior_results.seed.eq(seed) & prior_results.level.isin(LEVELS)]
        pd.testing.assert_frame_equal(pd.DataFrame(rows).groupby("level").sum(),
            archived.set_index("level")[["tp","fp","fn","tn"]].sort_index())
        print(f"Archived counts verified for seed {seed}", flush=True)


def rolling_sum(a, width):
    return np.convolve(a, np.ones(width), "same")


def raw_windows(ts, raw, state):
    good = np.isfinite(raw[:,:2]).all(axis=1) & (state == 2)
    energy = np.where(good, np.square(raw[:,:2].astype(float)).sum(axis=1), 0)
    rms = np.sqrt(rolling_sum(energy,7)/7)
    ids = np.arange(10, len(raw)-3)
    sec = ts.astype(np.int64)
    complete = (rolling_sum(good,7) == 7)
    c = np.arange(3,len(raw)-3)
    complete[c] &= sec[c+3] - sec[c-3] == 6
    ids = ids[complete[ids] & complete[ids-7] & (sec[ids]-sec[ids-7] == 7) & (rms[ids-7]>0)]
    changes = np.abs(rms[ids]/rms[ids-7]-1)*100
    assert len(changes)>100
    return rms, complete, float(np.quantile(changes, .9)), len(changes)


def local(plan_dir, fits, bundle, out):
    ph = replay.sha(plan_dir / "plan.json")
    plan = read_json(plan_dir / "plan.json")
    config = read_json(bundle / "config.json")
    assert ph == config["plan_sha"]
    candidate = config["candidate"]
    name = next(iter(candidate))
    sys.path.insert(0, str(replay.LOCK))
    mod, _ = replay.load_locked()
    mod.DEVICE = torch.device("cpu")
    torch.set_num_threads(3)
    dates, cases, reference_rows = {}, {}, []
    first_fit = fits / "seed_42/reference.npz"
    assert replay.sha(first_fit) == plan["fits"]["42"]["reference"]
    with np.load(first_fit) as a:
        mean42, scale42 = a["mean"], a["scale"]
    for day in plan["test"]:
        ts, raw, state = replay.read_day(day)
        prep = replay.DAILY / f"{day}_preprocessed.csv"
        lab = replay.LABELS / f"{day}_labels.csv"
        for source in (prep,lab):
            assert replay.sha(source) == plan["hashes"][source.relative_to(replay.ROOT).as_posix()]
        t0, initial = fusion.load_background(prep)
        np.testing.assert_array_equal(ts, t0)
        maskfile = plan_dir / f"{day}.npz"
        assert replay.sha(maskfile) == plan["masks"][day]
        with np.load(maskfile) as a:
            for x,y in [(ts,a["timestamp"]),(state,a["state"]),(np.isfinite(raw),a["observed"])]:
                np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(raw[np.isfinite(raw)], initial[np.isfinite(raw)])
        z0, valid0 = ctx.context(ts, raw, mean42, scale42)
        base_rms, complete, q90, n_ref = raw_windows(ts, raw, state)
        reference_rows.append(dict(date=day, q90_abs_xy_rms_change_pct=q90, pairs=n_ref))
        dates[day] = (ts, raw, z0, valid0)
        for level in LEVELS:
            for pct in replay.FRACTIONS:
                key = f"{day}_imbalance_pct{pct}_sev{level}"
                for suffix in ("data","labels"):
                    source = replay.SCENARIOS / f"{key}_{suffix}.parquet"
                    assert replay.sha(source) == plan["hashes"][source.relative_to(replay.ROOT).as_posix()]
                a, label = replay.read_scenario(key,ts,state)
                masked = np.where(np.isfinite(raw),a,np.nan)
                _, valid = ctx.context(ts,masked,mean42,scale42)
                keep = valid & (label | (state==2))
                fault_centers = np.flatnonzero(keep & label)
                clean7 = complete & (rolling_sum(np.isfinite(masked[:,:2]).all(axis=1),7)==7) & (base_rms>0)
                select = clean7[fault_centers] & valid0[fault_centers]
                pos = fault_centers[select]
                mr = np.nan_to_num(masked[:,:2].astype(float))
                rr = np.nan_to_num(raw[:,:2].astype(float))
                a_rms = np.sqrt(rolling_sum(np.square(mr).sum(1),7)/7)
                d_rms = np.sqrt(rolling_sum(np.square(mr-rr).sum(1),7)/7)
                level_change = 100*(a_rms[pos]/base_rms[pos]-1)
                frame = pd.DataFrame(dict(scenario=key,date=day,level=level,pct=pct,ts=ts[pos],
                    base_x=raw[pos,0],base_y=raw[pos,1],base_z=raw[pos,2],base_t=raw[pos,3],
                    injected_x=a[pos,0],injected_y=a[pos,1],injected_z=a[pos,2],injected_t=a[pos,3],
                    xy_rms_background=base_rms[pos],xy_rms_injected=a_rms[pos],
                    xy_rms_change_pct=level_change,abs_xy_change_pct=np.abs(level_change),
                    xy_perturbation_pct=100*d_rms[pos]/base_rms[pos],natural_q90_pct=q90,
                    above_natural_q90=np.abs(level_change)>q90,full_fault7=rolling_sum(label,7)[pos]==7))
                cases[key] = (day, keep, label[keep], select, pos, frame)
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(reference_rows).to_csv(out/"natural_reference.csv",index=False)
    audits = []
    for seed in replay.SEEDS:
        dir_seed = bundle/str(seed)
        with np.load(dir_seed/"cal.npz") as a:
            score = build_scorer(a["x"],a["z"],config["incumbent"],candidate)
        fit = fits / ("seed_42" if seed == 42 else f"seed_{seed}_fit")
        assert replay.sha(fit/"reference.npz") == plan["fits"][str(seed)]["reference"]
        assert replay.sha(fit/"model.pt") == plan["fits"][str(seed)]["model"]
        with np.load(fit/"reference.npz") as a:
            ref = dict(a)
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = ref["mean"], ref["scale"]
        model = mod._build_ae("Conv1DAE",4,8,[],[]).to(mod.DEVICE)
        model.load_state_dict(torch.load(fit/"model.pt",map_location=mod.DEVICE,weights_only=True))
        bg_scores = {}
        for d,(ts,raw,z,valid) in dates.items():
            x = fusion.component_scores(mod,model,scaler,ref,ts,raw)
            np.testing.assert_array_equal(np.isfinite(x).all(1),valid)
            pred = np.full((len(ts),5),np.nan,dtype=np.float32)
            pred[valid] = score(x[valid],z[valid])
            bg_scores[d] = pred
        cutoff = np.float32(config["cutoffs"][str(seed)][name])
        frames = []
        for key,(day,_,label,select,pos,base) in cases.items():
            with np.load(dir_seed/f"{key}.npz") as a:
                np.testing.assert_array_equal(a["y"],label)
                val = a["positive"]
            assert len(val) == len(select)
            hit = val[:,0] > cutoff
            val, bg = val[select], bg_scores[day][pos]
            assert np.isfinite(bg).all()
            frame = base.assign(seed=seed, hit=hit[select], cutoff=cutoff,
                score_injected=val[:,0],score_background=bg[:,0],margin=val[:,0]-cutoff,
                delta_score=val[:,0]-bg[:,0],background_hit=bg[:,0]>cutoff)
            for i,part in enumerate(PARTS,1):
                frame[f"delta_{part}"] = val[:,i]-bg[:,i]
            frames.append(frame)
            audits.append(dict(seed=seed,scenario=key,eligible_positive=len(hit),
                archived_hit=int(hit.sum()),paired=len(val),paired_hit=int(hit[select].sum())))
        pd.concat(frames,ignore_index=True).to_parquet(out/f"pairs_seed{seed}.parquet",index=False)
        print(f"Seed {seed}: joined {sum(map(len,frames))} paired windows",flush=True)
    pd.DataFrame(audits).to_csv(out/"pairing_audit.csv",index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    commands = p.add_subparsers(dest="command", required=True)
    for name, args in (("export", ("root","out")), ("local", ("plan_dir","fits","bundle","out"))):
        s = commands.add_parser(name)
        for arg in args:
            s.add_argument("--"+arg.replace("_","-"),type=Path,required=True)
    a = p.parse_args()
    if a.command == "export":
        export(a.root,a.out)
    else:
        local(a.plan_dir,a.fits,a.bundle,a.out)

