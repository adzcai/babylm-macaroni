#!/usr/bin/env python3
"""Figs 4 & 6 — word-level cross-lingual alignment (static ``wte``), in
probability-of-superiority (PoS) units.

PoS = mean over single-token translation pairs of ``pct_raw`` (the percentile of
the pair's cosine within its frequency-matched random cross-lingual null =
P(translation pair is more similar than a random frequency-matched pair)).
Chance = 0.5. Ported from ``wordlevel_pos_plot.py``.

Outputs (into ``--out``, default ``out/figures``):
  * ``fig04_06_wordlevel.csv``             per-pair PoS (compute cache; role/tier/run)
  * ``fig04_06_wordlevel.png``             Fig 4 body bars: curriculum CS vs unilingual
  * ``fig04_06_wordlevel_shuffled.png``    appendix bars: shuffled CS vs unilingual
  * ``fig04_06_wordlevel_trajectory.png``  Fig 6 curriculum trajectory (if the
                                           per-stage checkpoints are available)

Run:  ``python figures/fig04_06_wordlevel.py [--seed 42] [--skip-compute]``

MODEL SET (public conditions; old run-name map in parentheses):
  curriculum (cur_cs) / curriculum_noswitch (cur_nsw)  -> Fig 4 body + Fig 6 traj
  switch     (rnd_cs) / noswitch            (rnd_nsw)   -> shuffled-bars appendix

⚠️ Fig 6 TRAJECTORY CHECKPOINT GAP: the public model repo publishes only each
run's FINAL checkpoint (one HF branch per condition). The intermediate
per-stage / per-epoch curriculum checkpoints the trajectory needs are NOT on the
Hub. This script discovers them under ``models/<condition>/stage{1,2,3}/checkpoint-*``
if a user has trained/placed them locally; if absent it prints a clear note and
skips the trajectory panel (the bar figures still render).
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import common` / `_lib`
import common
from common import plt, MUT, INK, TIER, style_axes, save, model_dir, final_ckpt
from _lib import align

SEED_DEFAULT = 42
# tier -> legend label (dark teal switched / light teal novel)
TIERS = {"attested": "switched pairs", "pair_unseen": "novel pairings"}

# (condition, freq_cond, display label). freq_cond selects the within-corpus
# frequency-matched null (cs models -> intra, unilingual -> noswitch).
FINAL_MODELS = [
    ("curriculum",          "intra",    "Curriculum\ncode-switched"),
    ("curriculum_noswitch", "noswitch", "Curriculum\nunilingual"),
    ("switch",              "intra",    "Shuffled\ncode-switched"),
    ("noswitch",           "noswitch", "Shuffled\nunilingual"),
]
BODY = ["curriculum", "curriculum_noswitch"]
SHUFFLED = ["switch", "noswitch"]
TRAJ_ARMS = [("curriculum", "intra"), ("curriculum_noswitch", "noswitch")]
STAGE_NAME = {"s1": "stage 1 · word-level CS", "s2": "stage 2 · sentence-level CS",
              "s3": "stage 3 · unilingual"}


# --------------------------------------------------------------------------- #
# Trajectory checkpoint discovery (graceful when absent — see module docstring).
# --------------------------------------------------------------------------- #
def _ckpt_step(p):
    m = re.search(r"checkpoint-(\d+)", p.name)
    return int(m.group(1)) if m else -1


def discover_traj_checkpoints(condition, seed, models_dir):
    """Return ordered [(stage, step, ckpt_dir)] for a curriculum run, incl the
    shared init (stage1/checkpoint-0). Raises FileNotFoundError if none exist."""
    base = model_dir(condition, seed, models_dir)
    found = []
    for si, stage in enumerate(("stage1", "stage2", "stage3"), start=1):
        sdir = base / stage
        if not sdir.is_dir():
            continue
        for ck in sorted(sdir.glob("checkpoint-*"), key=_ckpt_step):
            if (ck / "model.safetensors").is_file():
                found.append((f"s{si}", _ckpt_step(ck), ck))
    if not found:
        raise FileNotFoundError(
            f"no per-stage checkpoints under {base}/stage*/checkpoint-* for "
            f"'{condition}'. The Fig 6 trajectory needs intermediate curriculum "
            f"checkpoints, which are NOT published on the Hub (only final models "
            f"are). Train locally or place stageN/checkpoint-* dirs to enable it.")
    return found


# --------------------------------------------------------------------------- #
# COMPUTE
# --------------------------------------------------------------------------- #
def compute(models_dir, data_dir, out_dir, seed):
    tok = common.load_tokenizer(model_dir("curriculum", seed, models_dir))
    lex, cands, cfreq = align.prepare_lexicon(
        data_dir, tok, cache_dir=out_dir / "align_cache")

    # final-checkpoint pairs for all four models
    final_labels = [(cond, fc, "final", final_ckpt(model_dir(cond, seed, models_dir)))
                    for cond, fc, _ in FINAL_MODELS]
    fin = align.wordlevel_pairs(final_labels, lex, cands, cfreq, seed=seed)
    fin["role"], fin["stage"], fin["step"] = "final", "", -1

    frames = [fin[["run", "role", "ckpt", "stage", "step", "tier", "pct_raw"]]]

    # trajectory (curriculum arms only; skip gracefully if checkpoints absent)
    for cond, fc in TRAJ_ARMS:
        try:
            cks = discover_traj_checkpoints(cond, seed, models_dir)
        except FileNotFoundError as e:
            print(f"[trajectory] {e}\n[trajectory] skipping trajectory for {cond}.")
            continue
        labels = [(cond, fc, f"{stage}-{step}", ck) for stage, step, ck in cks]
        tj = align.wordlevel_pairs(labels, lex, cands, cfreq, seed=seed)
        step_of = {f"{s}-{st}": (s, st) for s, st, _ in cks}
        tj["role"] = "traj"
        tj["stage"] = tj["ckpt"].map(lambda c: step_of[c][0])
        tj["step"] = tj["ckpt"].map(lambda c: step_of[c][1])
        frames.append(tj[["run", "role", "ckpt", "stage", "step", "tier", "pct_raw"]])

    out = pd.concat(frames, ignore_index=True)
    csv = out_dir / "fig04_06_wordlevel.csv"
    out.to_csv(csv, index=False)
    print(f"wrote {csv}")
    return out


# --------------------------------------------------------------------------- #
# PLOT
# --------------------------------------------------------------------------- #
def _final_agg(df):
    """(run, tier) -> mean PoS over pairs + per-tier n. One --seed invocation =
    one checkpoint per model, so there is no seed-replication spread to draw as
    an error bar (unlike the 3-seed paper suite); bars are drawn without yerr."""
    f = df[(df.role == "final") & (df.tier.isin(TIERS))]
    agg = f.groupby(["run", "tier"])["pct_raw"].mean()
    n_by = f.groupby(["run", "tier"]).size().groupby("tier").max()
    return agg, n_by


def fig_bars(df, conditions, labels, out):
    agg, n_by = _final_agg(df)
    fig, ax = plt.subplots(figsize=(4.3, 3.7))
    ax.axhline(0.5, color=MUT, lw=1, ls=":", zorder=1)
    ax.set_xlim(-0.6, len(conditions) - 0.18)
    ax.text(len(conditions) - 0.3, 0.5, "chance", fontsize=7.5, color=MUT,
            va="center", ha="left")
    width = 0.38
    for j, tier in enumerate(TIERS):
        xs = [i + (j - 0.5) * width for i in range(len(conditions))]
        m = [agg.loc[(c, tier)] for c in conditions]
        bars = ax.bar(xs, m, width=width, color=TIER[tier], zorder=3,
                      edgecolor="white", lw=0.6,
                      label=f"{TIERS[tier]} (n={int(n_by[tier]):,})")
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("probability of superiority\n(final checkpoint)", color=INK, fontsize=9)
    ax.set_ylim(0.4, 1.03)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    save(fig, out)
    print(f"wrote {out.name}")


def fig_trajectory(df, out):
    tj = df[(df.role == "traj") & (df.tier.isin(TIERS))]
    if tj.empty:
        print("[trajectory] no trajectory data (checkpoints absent); skipping "
              "fig04_06_wordlevel_trajectory.png")
        return
    # ordered checkpoint grid: unique (stage, step) sorted by stage then step
    grid = (tj[["stage", "step"]].drop_duplicates()
            .sort_values(["stage", "step"]).reset_index(drop=True))
    grid["x"] = range(len(grid))
    xpos = {(s, st): x for s, st, x in zip(grid.stage, grid.step, grid.x)}
    # single --seed => mean PoS per checkpoint, no seed band (see _final_agg note)
    agg = (tj.groupby(["run", "tier", "stage", "step"])["pct_raw"]
           .mean().reset_index())
    agg["x"] = [xpos[(s, st)] for s, st in zip(agg.stage, agg.step)]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhline(0.5, color=MUT, lw=1, ls=":", zorder=1)
    ax.text(len(grid) - 0.4, 0.5, "chance", fontsize=8, color=MUT, va="center")
    arms = [("curriculum_noswitch", "--", "unilingual"),
            ("curriculum", "-", "code-switched")]
    for run, ls, mlab in arms:
        for tier in TIERS:
            t = agg[(agg.run == run) & (agg.tier == tier)].sort_values("x")
            if t.empty:
                continue
            ax.plot(t.x, t["pct_raw"], ls, marker="o", color=TIER[tier], lw=2.2, ms=4,
                    label=f"{mlab} · {TIERS[tier]}")
    # stage bands
    for stage, g in grid.groupby("stage"):
        a, b = g.x.min(), g.x.max()
        if stage != "s3":
            ax.axvspan(a - 0.5, b + 0.5, color="#dce9f7" if stage == "s1" else "#e8f3e3",
                       alpha=0.6, lw=0)
        ax.text((a + b) / 2, 1.015, STAGE_NAME.get(stage, stage),
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color=MUT)
    ax.set_xticks(list(grid.x))
    ax.set_xticklabels([str(s) for s in grid.step], fontsize=8.5)
    ax.set_xlabel("checkpoint step within stage", color=MUT)
    ax.set_ylabel("probability of superiority\n(translation pair beats frequency-matched null)",
                  color=INK)
    ax.set_xlim(-0.5, len(grid) - 0.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    save(fig, out)
    print(f"wrote {out.name}")


def plot(df, out_dir):
    body_labels = [lab for c, _, lab in FINAL_MODELS if c in BODY]
    shuf_labels = [lab for c, _, lab in FINAL_MODELS if c in SHUFFLED]
    fig_bars(df, BODY, body_labels, out_dir / "fig04_06_wordlevel.png")
    fig_bars(df, SHUFFLED, shuf_labels, out_dir / "fig04_06_wordlevel_shuffled.png")
    fig_trajectory(df, out_dir / "fig04_06_wordlevel_trajectory.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--skip-compute", action="store_true",
                    help="replot from an existing fig04_06_wordlevel.csv")
    args = ap.parse_args()
    models_dir, data_dir, out_dir = common.resolve_paths(args)

    csv = out_dir / "fig04_06_wordlevel.csv"
    if args.skip_compute:
        if not csv.is_file():
            raise SystemExit(f"--skip-compute but {csv} does not exist; run compute first.")
        df = pd.read_csv(csv)
    else:
        df = compute(models_dir, data_dir, out_dir, args.seed)
    plot(df, out_dir)


if __name__ == "__main__":
    main()
