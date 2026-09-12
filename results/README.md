# results/ — the paper's measurement CSVs

These are the exact inputs the paper's figures were plotted from (8 seeds, 42–49), with the
internal run names replaced by the public condition names (`curriculum`, `curriculum_noswitch`,
`switch`, `noswitch`, `word`, `sent`, `par`, `salad`; seed 42 bare, other seeds `-sNN`).
`figures/make_all.py` plots them as-is; `figures/measure_*.py` regenerate them from models.

| File | Rows | Produced by | Read by |
|---|---|---|---|
| `retrieval_finals.csv` | model, group, seed, layer, pair, retrieval metrics | `measure_retrieval.py --set finals` | fig2, fig6, fig7 |
| `retrieval_controls.csv` | same, for the four controls | `measure_retrieval.py --set controls` | fig7 |
| `retrieval_trajectory.csv` | same, model = `cur_{cs,noswitch}-s<seed>-s<stage>e<epoch>` | `measure_retrieval.py --set trajectory` | fig3 |
| `exposure_rank.csv` | seed, subset, group, direction, stage, epoch, delta, ci, medians, n | `measure_wordlevel.py` | fig4 |
| `word_divergence.csv` | lang, condition, layer, n_pairs, emb/ctl cosine + L2, d, p, seed | `measure_wordlevel.py` | fig8 |

Retrieval metrics: `retrieval_p1` (CSLS k=10 bitext retrieval P@1, both directions averaged),
`retrieval_p1_nn` (plain cosine), `median_pct_rank`, `mean_pct_rank`, `mrr`, `retrieval_p5`,
`median_rank`, `cos_gap`, `cka`, `centroid_dist`; `group` is `switched` / `noswitch`.
