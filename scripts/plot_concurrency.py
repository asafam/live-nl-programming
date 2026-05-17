#!/usr/bin/env python3
"""Plot concurrent-events pass rate across concurrency levels and paradigms.

Usage:
    python scripts/plot_concurrency.py [exp_dir] [--smooth N]

    exp_dir   — directory containing exp_*.jsonl files
                (default: outputs/data/zapier/runs/experiments/concurrency)
    --smooth N — interpolate each line through N points using a cubic spline
                 so segments curve instead of being straight (default: 0 = off)

Reads files matching: exp_{lnl|baseline}_{N}mod_conc{C}[_{single|multi}].jsonl
Generates plots saved to <exp_dir>/plots/:
  - concurrency_passrate[__<judge>].png  — one per judge found

Three series:
  - Ours (LNL):              exp_lnl_*  files
  - OpenClaw (single-agent): exp_baseline_*_single.jsonl files
  - OpenClaw (multi-agent):  exp_baseline_*_multi.jsonl  files
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ── Metric definitions ────────────────────────────────────────────────────────

PASS_METRICS = [
    ("mean",       "Mean pass rate"),
    ("steps",      "Steps pass rate"),
    ("mod",        "Mod pass rate (pre+post)"),
    ("pre_mod",    "Pre-mod pass rate"),
    ("post_mod",   "Post-mod pass rate"),
    ("irrelevant", "Irrelevant pass rate"),
]

STEP_ID = re.compile(r"^S\d+$")

# Series: (paradigm, agent_mode) → display label, color
SERIES_LABEL = {
    ("lnl",      "lnl"):    "Ours",
    ("baseline", "single"): "OpenClaw (single-agent)",
    ("baseline", "multi"):  "OpenClaw (multi-agent)",
}
SERIES_COLOR = {
    ("lnl",      "lnl"):    "#2196F3",  # blue
    ("baseline", "single"): "#E64A19",  # orange-red
    ("baseline", "multi"):  "#4CAF50",  # green
}
MOD_LINESTYLE = {1: "-",  2: "--"}
MOD_MARKER    = {1: "o",  2: "s"}
MOD_ALPHA     = {1: 1.0,  2: 0.75}

FILL_ALPHA = 0.12  # gradient fill below each line

_ORIGINAL = "__original__"


# ── File scanning ─────────────────────────────────────────────────────────────

def _scan_judges(path: Path) -> tuple[str, list[str]]:
    original = "unknown"
    rejudge_models: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "tc_id" not in d:
                original = (d.get("judge_model")
                            or (d.get("params") or {}).get("judge_model")
                            or original)
                continue
            for e in d.get("events", []):
                for rj in e.get("rejudges", []):
                    if rj.get("model"):
                        rejudge_models.add(rj["model"])
    return original, sorted(rejudge_models)


# ── Metric computation ────────────────────────────────────────────────────────

def _get_passed_original(evt: dict):
    return evt.get("passed")


def _make_get_passed_rejudge(model: str):
    def _get(evt: dict):
        for rj in evt.get("rejudges", []):
            if rj.get("model") == model:
                return rj.get("passed")
        return None
    return _get


def compute_metrics(path: Path, get_passed=_get_passed_original) -> dict:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "tc_id" not in d:
                continue
            results.append(d)

    if not results:
        return {}

    first_tc_per_sample: dict[str, str] = {}
    for r in results:
        sid = r.get("sample_id") or r["tc_id"]
        if sid not in first_tc_per_sample:
            first_tc_per_sample[sid] = r["tc_id"]
    base_tc_ids = set(first_tc_per_sample.values())

    by_role: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    pass_rates: list[float] = []

    for r in results:
        is_base = r["tc_id"] in base_tc_ids
        events = r.get("events", [])

        effective = [e for e in events if is_base or not STEP_ID.match(e["event_id"])]
        scored = [e for e in effective if get_passed(e) is not None]
        if scored:
            pass_rates.append(sum(1 for e in scored if get_passed(e)) / len(scored))

        for e in events:
            eid = e["event_id"]
            role = e.get("role")
            if STEP_ID.match(eid):
                if not is_base:
                    continue
                role = "step"
            elif not role:
                continue
            p = get_passed(e)
            if p is None:
                continue
            by_role[role]["pass"]  += 1 if p else 0
            by_role[role]["total"] += 1

    def rate(role: str):
        s = by_role.get(role)
        return s["pass"] / s["total"] if s and s["total"] else None

    def mean(xs): return sum(xs) / len(xs) if xs else None

    mod_pass  = sum(by_role[r]["pass"]  for r in ("pre_mod", "post_mod"))
    mod_total = sum(by_role[r]["total"] for r in ("pre_mod", "post_mod"))

    return {
        "mean":       mean(pass_rates),
        "steps":      rate("step"),
        "mod":        mod_pass / mod_total if mod_total else None,
        "pre_mod":    rate("pre_mod"),
        "post_mod":   rate("post_mod"),
        "irrelevant": rate("irrelevant"),
        "n_tcs":      len(results),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_experiments(exp_dir: Path) -> tuple[dict, dict]:
    """Return (data_by_judge, original_judge_name).

    data_by_judge: {judge_key: {(paradigm, agent_mode, mods): {conc: metrics}}}
    """
    pattern = re.compile(
        r"exp_(lnl|baseline)_(\d+)mod_conc(\d+)(?:_(single|multi))?\.jsonl$"
    )
    data_by_judge: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    original_names: dict[str, str] = {}

    paths = sorted(exp_dir.glob("exp_*.jsonl"))
    for path in paths:
        m = pattern.match(path.name)
        if not m:
            continue
        paradigm = m.group(1)
        mods     = int(m.group(2))
        conc     = int(m.group(3))
        suffix   = m.group(4)  # "single", "multi", or None

        if paradigm == "lnl":
            agent_mode = "lnl"
        elif suffix in ("single", "multi"):
            agent_mode = suffix
        else:
            agent_mode = "single"

        series_key = (paradigm, agent_mode, mods)

        orig_model, rejudge_models = _scan_judges(path)
        original_names[_ORIGINAL] = original_names.get(_ORIGINAL, orig_model)

        metrics = compute_metrics(path, _get_passed_original)
        if not metrics:
            print(f"  Skipping {path.name} (empty)")
            continue
        data_by_judge[_ORIGINAL][series_key][conc] = metrics
        mean_str = f"{metrics['mean']:.1%}" if metrics.get("mean") is not None else "N/A"
        print(f"  {path.name}: {metrics['n_tcs']} TCs  mean={mean_str} (judge: {orig_model})")

        for rj_model in rejudge_models:
            get_passed = _make_get_passed_rejudge(rj_model)
            rj_metrics = compute_metrics(path, get_passed)
            if rj_metrics:
                data_by_judge[rj_model][series_key][conc] = rj_metrics
                rj_mean = f"{rj_metrics['mean']:.1%}" if rj_metrics.get("mean") is not None else "N/A"
                print(f"    └─ rejudge {rj_model}: mean={rj_mean}")

    return dict(data_by_judge), original_names


# ── Plotting ──────────────────────────────────────────────────────────────────

def _smooth_xy(x: list, y: list, n_points: int) -> tuple:
    """Return (xs, ys) interpolated through (x, y) with a cubic spline.

    Falls back to the raw arrays when there are fewer than 3 finite points
    (not enough for a cubic spline).
    """
    try:
        from scipy.interpolate import make_interp_spline
    except ImportError:
        return x, y

    pairs = [(xi, yi) for xi, yi in zip(x, y) if yi == yi]  # drop NaN
    if len(pairs) < 3:
        return x, y

    xs_raw = [p[0] for p in pairs]
    ys_raw = [p[1] for p in pairs]
    xs_fine = np.linspace(xs_raw[0], xs_raw[-1], n_points)
    spline = make_interp_spline(xs_raw, ys_raw, k=min(3, len(pairs) - 1))
    ys_fine = np.clip(spline(xs_fine), 0.0, 1.0)
    return xs_fine, ys_fine


def _draw_lines(ax, data: dict, metric_key: str, all_concs: list,
                smooth: int = 0) -> bool:
    plotted = False
    multi_mods = len({k[2] for k in data}) > 1
    for series_key, series in sorted(data.items()):
        paradigm, agent_mode, mods = series_key
        label_base = SERIES_LABEL.get((paradigm, agent_mode))
        if label_base is None:
            continue
        concs  = sorted(series)
        values = [series[c].get(metric_key) for c in concs]
        if all(v is None for v in values):
            continue
        y_raw = [v if v is not None else float("nan") for v in values]

        color     = SERIES_COLOR.get((paradigm, agent_mode), "#888888")
        linestyle = MOD_LINESTYLE.get(mods, "-")
        marker    = MOD_MARKER.get(mods, "o")
        alpha     = MOD_ALPHA.get(mods, 1.0)
        label     = f"{label_base} ({mods} mod{'s' if mods > 1 else ''})" if multi_mods else label_base

        if smooth > 0 and len(concs) >= 3:
            x_plot, y_plot = _smooth_xy(concs, y_raw, smooth)
            # Draw the smooth curve (no markers on the interpolated line)
            ax.plot(x_plot, y_plot, linestyle=linestyle, color=color,
                    alpha=alpha, linewidth=2, label=label)
            # Overlay actual data points on top
            ax.plot(concs, y_raw, linestyle="none", marker=marker,
                    color=color, alpha=alpha, markersize=7, zorder=5)
            ax.fill_between(x_plot, y_plot, 0, color=color,
                            alpha=FILL_ALPHA * alpha)
        else:
            ax.plot(concs, y_raw, linestyle=linestyle, marker=marker,
                    color=color, alpha=alpha, label=label,
                    linewidth=2, markersize=7)
            ax.fill_between(concs, y_raw, 0, color=color,
                            alpha=FILL_ALPHA * alpha)

        plotted = True

    if all_concs:
        ax.set_xticks(all_concs)
    ax.set_xlabel("Concurrency level", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    if plotted:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    return plotted


def _save(fig, path: Path, title: str) -> None:
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def _filename_safe(model: str) -> str:
    return re.sub(r"[^\w\-.]", "_", model)


def plot_passrate(data: dict, plots_dir: Path, judge_label: str,
                  filename_suffix: str = "", smooth: int = 0) -> None:
    all_concs = sorted({c for s in data.values() for c in s})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (key, label) in zip(axes.flatten(), PASS_METRICS):
        _draw_lines(ax, data, key, all_concs, smooth=smooth)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Pass rate", fontsize=9)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(-0.02, 1.05)
    fname = "concurrency_passrate"
    if filename_suffix:
        fname += f"__{filename_suffix}"
    fname += ".png"
    _save(fig, plots_dir / fname,
          f"Ours vs OpenClaw — Concurrency Pass Rates  (judge: {judge_label})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_dir", nargs="?",
                        default="outputs/data/zapier/runs/experiments/concurrency",
                        help="Directory containing exp_*.jsonl files")
    parser.add_argument("--smooth", type=int, default=0, metavar="N",
                        help="Interpolate lines with a cubic spline through N points "
                             "(0 = off, straight segments; 300 = smooth curves). "
                             "Requires scipy.")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Directory not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading experiments from {exp_dir} ...")
    data_by_judge, original_names = load_experiments(exp_dir)

    if not data_by_judge:
        print("No matching experiment files found.", file=sys.stderr)
        sys.exit(1)

    plots_dir = exp_dir / "plots"
    all_judge_keys = sorted(data_by_judge)
    multi_judge = len(all_judge_keys) > 1

    print(f"\nGenerating plots → {plots_dir}")

    for judge_key in all_judge_keys:
        data = data_by_judge[judge_key]
        if judge_key == _ORIGINAL:
            display = original_names.get(_ORIGINAL, "original")
            suffix = _filename_safe(display) if multi_judge else ""
        else:
            display = judge_key
            suffix = _filename_safe(judge_key)
        plot_passrate(data, plots_dir, judge_label=display,
                      filename_suffix=suffix, smooth=args.smooth)

    print("Done.")


if __name__ == "__main__":
    main()
