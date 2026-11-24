"""Fit one locked Exp3 revision seed on a host with the archived fit data."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

import evaluate_exp3_revision_alarm as replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=replay.SEEDS, required=True)
    args = parser.parse_args()
    os.environ["EXP3_ROOT"] = str(args.root.resolve())
    replay.ROOT = args.root
    replay.LOCK = args.lock
    replay.DAILY = args.root / "data/monitoring/line_1"
    replay.LABELS = args.root / "data/labels/line_1"
    replay.SCENARIOS = args.root / "data/scenarios/line_1"
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if replay.sha(replay.LOCK / "config.py") != manifest["config_sha256"]:
        raise ValueError("Locked config does not match seed 42")
    if replay.sha(replay.LOCK / "exp3_model_selection.py") != manifest["model_driver_sha256"]:
        raise ValueError("Model driver does not match seed 42")
    if replay.sha(Path(replay.__file__)) != manifest["driver_sha256"]:
        raise ValueError("Fit driver does not match seed 42")
    for rel, digest in manifest["train_input_sha256"].items():
        if replay.sha(args.root / rel.replace("\\", "/")) != digest:
            raise ValueError(f"Training data hash mismatch: {rel}")
    sys.path.insert(0, str(args.lock))
    mod, _ = replay.load_locked()
    print("Verified all training inputs against seed 42", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    torch.set_num_threads(4)
    mod._set_device(0)
    args.out.mkdir(parents=True, exist_ok=False)
    model, scaler, latent, residual, stat, loss = replay.fit(mod, manifest["train"], args.seed)
    torch.save(model.state_dict(), args.out / "model.pt")
    np.savez(args.out / "reference.npz", mean=scaler.mean_, scale=scaler.scale_,
             latent=latent, residual=residual, stats=stat)
    manifest.pop("checkpoint", None)
    manifest.pop("checkpoint_sha256", None)
    manifest.update(seed=args.seed, python=sys.version, torch=str(torch.__version__),
                    device=torch.cuda.get_device_name(0), fit_driver_sha256=replay.sha(Path(__file__)))
    (args.out / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Seed {args.seed}: saved locked checkpoint and reference; loss={loss}", flush=True)


if __name__ == "__main__":
    main()
