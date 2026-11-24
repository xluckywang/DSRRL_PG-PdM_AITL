"""Small plotting helpers used by the public EXP3 figure script."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from matplotlib import pyplot as plt


@contextmanager
def publication_context(font_family: str = "Times New Roman", base_size: float = 8):
    settings = {
        "font.family": font_family,
        "font.size": base_size,
        "axes.titlesize": base_size,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 0.5,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
    }
    with plt.rc_context(settings):
        yield settings


def export_figure(fig, stem: Path, dpi: int = 600, provenance: dict | None = None) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    if provenance is not None:
        stem.with_suffix(".manifest.json").write_text(
            json.dumps(provenance, indent=2, default=str), encoding="utf-8"
        )
