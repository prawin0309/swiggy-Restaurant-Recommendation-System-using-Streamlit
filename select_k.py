"""Choose the number of KMeans clusters with the elbow method.

Sweeps ``k`` over ``K_RANGE``, fits MiniBatchKMeans on the full encoded matrix
once per seed, and averages the inertia across seeds so the curve is smooth
enough for an elbow to be meaningful - a single MiniBatch run is stochastic
enough that inertia can rise with k, which makes the elbow unreadable.

The elbow is located with the Kneedle criterion: normalise the curve to the
unit square, draw the chord from the first point to the last, and take the k
whose vertical distance below that chord is greatest.

Run::

    python select_k.py

Writes ``reports/k_selection.json`` and ``reports/figures/07_elbow.png`` and
prints the k to put in ``config.N_CLUSTERS``.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

import config
from data_pipeline import load_datasets

K_RANGE = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32)
SEEDS = (42, 43, 44)
SILHOUETTE_SAMPLE = 5000
REPORT_DIR = config.BASE_DIR / "reports"
FIG_DIR = REPORT_DIR / "figures"


def _fit(encoded, k: int, seed: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=k, random_state=seed, n_init=10,
        batch_size=4096, max_iter=500,
    ).fit(encoded)


def _elbow(ks: list[int], inertia: list[float]) -> int:
    """Kneedle: k of maximum vertical drop below the first-to-last chord."""
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertia, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min())
    y_n = (y - y.min()) / (y.max() - y.min())
    # Chord from (0, 1) to (1, 0) for a decreasing curve; the elbow is the
    # point furthest BELOW it.
    chord = 1.0 - x_n
    return int(x[np.argmax(chord - y_n)])


def main() -> int:
    cleaned, encoded, columns = load_datasets()
    print(f"[data] encoded={encoded.shape}  nnz={encoded.nnz:,}  "
          f"density={encoded.nnz / (encoded.shape[0] * encoded.shape[1]):.4%}")

    rng = np.random.default_rng(config.RANDOM_SEED)
    sample = rng.choice(encoded.shape[0], SILHOUETTE_SAMPLE, replace=False)
    sampled = encoded[sample]

    rows = []
    print(f"\n{'k':>4} {'inertia (mean)':>16} {'spread':>9} "
          f"{'silhouette':>11} {'smallest':>9} {'largest':>9}")
    for k in K_RANGE:
        inertias, sils, smallest, largest = [], [], [], []
        for seed in SEEDS:
            model = _fit(encoded, k, seed)
            counts = np.bincount(model.labels_, minlength=k)
            inertias.append(float(model.inertia_))
            sils.append(float(silhouette_score(sampled, model.predict(sampled))))
            smallest.append(int(counts.min()))
            largest.append(int(counts.max()))
        row = {
            "k": k,
            "inertia": float(np.mean(inertias)),
            "inertia_spread": float(np.ptp(inertias)),
            "silhouette": float(np.mean(sils)),
            "smallest_cluster": int(np.mean(smallest)),
            "largest_cluster": int(np.mean(largest)),
        }
        rows.append(row)
        print(f"{k:>4} {row['inertia']:>16,.0f} {row['inertia_spread']:>9,.0f} "
              f"{row['silhouette']:>11.4f} {row['smallest_cluster']:>9,} "
              f"{row['largest_cluster']:>9,}")

    ks = [r["k"] for r in rows]
    inertia = [r["inertia"] for r in rows]
    elbow = _elbow(ks, inertia)

    # k=2 or 3 wins silhouette by splitting the catalogue into two enormous
    # blobs, which is useless as a browsing segmentation - so the reported
    # silhouette peak is taken over the usable range only.
    usable = [r for r in rows if r["k"] >= 4]
    best_sil = max(usable, key=lambda r: r["silhouette"])
    degenerate = [r for r in rows if r["k"] < 4]

    print(f"\n[elbow]      k = {elbow}  (Kneedle on mean inertia)")
    print(f"[silhouette] k = {best_sil['k']} at {best_sil['silhouette']:.4f} "
          f"over k>=4, where the whole range is "
          f"{min(r['silhouette'] for r in usable):.4f}-"
          f"{max(r['silhouette'] for r in usable):.4f} - too flat to choose on")
    for r in degenerate:
        print(f"[degenerate] k={r['k']} scores {r['silhouette']:.4f} but its "
              f"largest cluster holds {r['largest_cluster']:,} restaurants")
    print(f"[config]     set N_CLUSTERS = {elbow}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "k_selection.json").write_text(
        json.dumps({"rows": rows, "elbow_k": elbow,
                    "best_silhouette_k_usable": best_sil["k"],
                    "seeds": list(SEEDS),
                    "silhouette_sample": SILHOUETTE_SAMPLE}, indent=2),
        encoding="utf-8",
    )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4))
    ax_left.plot(ks, inertia, marker="o", color="#2b6cb0")
    ax_left.plot([ks[0], ks[-1]], [inertia[0], inertia[-1]],
                 linestyle="--", color="#a0aec0", label="chord")
    ax_left.axvline(elbow, color="#e53e3e", linestyle=":",
                    label=f"elbow k={elbow}")
    ax_left.set_title("Elbow — inertia vs k (mean of 3 seeds)")
    ax_left.set_xlabel("k")
    ax_left.set_ylabel("Inertia")
    ax_left.legend()

    ax_right.plot(ks, [r["silhouette"] for r in rows], marker="o",
                  color="#38a169")
    ax_right.axvline(elbow, color="#e53e3e", linestyle=":")
    ax_right.set_title("Silhouette vs k (5,000-row sample)")
    ax_right.set_xlabel("k")
    ax_right.set_ylabel("Silhouette")
    ax_right.set_ylim(0, max(0.12, max(r["silhouette"] for r in rows) * 1.3))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_elbow.png", dpi=150)
    plt.close(fig)
    print(f"[fig] {FIG_DIR / '07_elbow.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
