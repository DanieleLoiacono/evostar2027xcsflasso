"""Statistical analysis, figures and report (derived/ only; reads derived/runs.csv etc.).

Answers the campaign questions:
  Q1 parity of approximation quality      -> parity study, primary metric grid_mae_eps
  Q2 systematic differences               -> parity study, convergence / final error / size / stability metrics
  Q3 Constant vs NLMS vs RLS              -> parity study, within each implementation
  Q4 Lasso Batch / Lasso Online trade-offs -> pyext study only (never pooled with parity)
  Q5 practical importance                 -> equivalence margins (TOST via bootstrap CIs) next to the tests
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as ss

from . import stats as S
from .metrics import METRICS
from .runs import campaign_paths

PARITY_METRICS = ["grid_mae_eps", "n_macro", "grid_rmse_eps", "grid_maxae_eps", "frac_within_eps",
                  "final_online_mae_eps", "auc_online_mae_eps", "steps_to_eps", "late_online_std_eps",
                  "mean_width_frac", "n_params"]
PRIMARY = ["grid_mae_eps", "n_macro"]
PRED_METRICS = ["grid_mae_eps", "n_macro", "auc_online_mae_eps"]
EXT_METRICS = ["grid_mae_eps", "n_params", "n_macro", "frac_zero_slopes", "auc_online_mae_eps", "runtime_s"]
IMPL_LABEL = {"cxx": "C++ xcslib", "py": "Python xcsf"}
IMPL_COLOR = {"cxx": "#1f77b4", "py": "#d62728"}


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, float) or isinstance(v, np.floating):
        av = abs(v)
        if av != 0 and (av < 1e-3 or av >= 1e5):
            return f"{v:.2e}"
        return f"{v:.{nd}g}" if av >= 1 else f"{v:.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame, nd=3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out)


def _margin(metric: str, an: dict, ref_values: np.ndarray):
    fam = METRICS[metric][0]
    if fam == "error_eps":
        return an["equivalence_margin_error_eps_fraction"]
    if fam == "fraction":
        return an.get("equivalence_margin_fraction_abs", 0.05)
    if fam == "size":
        med = float(np.nanmedian(ref_values)) if len(ref_values) else float("nan")
        return an["equivalence_margin_size_relative"] * med if np.isfinite(med) and med > 0 else None
    return None


# ----------------------------------------------------------------------------------------- parity

def parity_tables(runs: pd.DataFrame, an: dict, rng: np.random.Generator) -> pd.DataFrame:
    par = runs[runs.study == "parity"]
    rows = []
    for (bench, arm), g in par.groupby(["benchmark", "arm"], sort=False):
        a_df = g[g.implementation == "py"].sort_values("run_id")
        b_df = g[g.implementation == "cxx"].sort_values("run_id")
        if a_df.empty or b_df.empty:
            continue
        for m in PARITY_METRICS:
            a, b = a_df[m].to_numpy(float), b_df[m].to_numpy(float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 2 or len(b) < 2:
                continue
            shift = S.hl_shift(a, b)
            ci = S.bootstrap_ci(S.hl_shift, [a, b], an["n_bootstrap"], rng)
            rows.append(dict(benchmark=bench, predictor=arm, metric=m, n_cxx=len(b), n_py=len(a),
                             median_cxx=float(np.median(b)), iqr_cxx=S.iqr(b), median_py=float(np.median(a)),
                             iqr_py=S.iqr(a), hl_shift_py_minus_cxx=shift, ci90_lo=ci[0.90][0], ci90_hi=ci[0.90][1],
                             ci95_lo=ci[0.95][0], ci95_hi=ci[0.95][1], p_mwu=S.mannwhitney_p(a, b),
                             a12_py_gt_cxx=S.a12(a, b), p_bf=S.brown_forsythe_p(a, b),
                             iqr_ratio_py_cxx=(S.iqr(a) / S.iqr(b)) if S.iqr(b) > 0 else float("nan"),
                             margin=_margin(m, an, b)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm"] = np.nan
    df["p_bf_holm"] = np.nan
    for m, g in df.groupby("metric"):
        df.loc[g.index, "p_holm"] = S.holm(g["p_mwu"].to_numpy())
        df.loc[g.index, "p_bf_holm"] = S.holm(g["p_bf"].to_numpy())
    df["effect"] = df["a12_py_gt_cxx"].map(S.a12_magnitude)
    df["verdict"] = [S.verdict(r.p_holm, (r.ci90_lo, r.ci90_hi), (r.ci95_lo, r.ci95_hi), r.hl_shift_py_minus_cxx,
                               r.margin, an["alpha"]) for r in df.itertuples()]
    return df


# ------------------------------------------------------------------------------------- predictors

def predictor_tables(runs: pd.DataFrame, an: dict):
    par = runs[runs.study == "parity"]
    omni, pair, ranks = [], [], []
    for (impl, bench), g in par.groupby(["implementation", "benchmark"], sort=False):
        arms = list(dict.fromkeys(g["arm"]))
        for m in PRED_METRICS:
            wide = g.pivot_table(index="run_id", columns="arm", values=m)[arms]
            med = wide.median()
            order = " < ".join(med.sort_values().index)
            if impl == "py":   # same seeds -> same training stream: paired design (blocks = run_id)
                w = wide.dropna()
                p = float(ss.friedmanchisquare(*[w[a] for a in arms]).pvalue) if len(arms) >= 3 and len(w) >= 2 else float("nan")
                test = "Friedman (paired by run)"
            else:              # xcslib draws inputs from its own RNG stream: treat as independent samples
                p = float(ss.kruskal(*[wide[a].dropna() for a in arms]).pvalue) if len(arms) >= 2 else float("nan")
                test = "Kruskal-Wallis"
            omni.append(dict(implementation=impl, benchmark=bench, metric=m, test=test, p=p, ordering_by_median=order,
                             **{f"median_{a}": float(med[a]) for a in arms}))
            ranks.append(dict(implementation=impl, benchmark=bench, metric=m, **{a: float(med[a]) for a in arms}))
            ps, recs = [], []
            for i, x in enumerate(arms):
                for y in arms[i + 1:]:
                    if impl == "py":
                        d = (wide[x] - wide[y]).dropna().to_numpy()
                        pv, sh = S.wilcoxon_p(d), S.hl_paired(d)
                    else:
                        pv, sh = S.mannwhitney_p(wide[x].dropna(), wide[y].dropna()), S.hl_shift(wide[x].dropna(), wide[y].dropna())
                    recs.append(dict(implementation=impl, benchmark=bench, metric=m, a=x, b=y, hl_shift_a_minus_b=sh,
                                     a12_a_gt_b=S.a12(wide[x].dropna(), wide[y].dropna()), p=pv))
                    ps.append(pv)
            adj = S.holm(ps)
            for r, pa in zip(recs, adj):
                r["p_holm"] = pa
                pair.append(r)
    omni = pd.DataFrame(omni)
    if not omni.empty:
        omni["p_holm"] = np.nan
        for (impl, m), g in omni.groupby(["implementation", "metric"]):
            omni.loc[g.index, "p_holm"] = S.holm(g["p"].to_numpy())
    # agreement of predictor rankings between implementations
    agree = []
    rk = pd.DataFrame(ranks)
    if not rk.empty and set(rk.implementation) >= {"cxx", "py"}:
        arms = [c for c in rk.columns if c not in ("implementation", "benchmark", "metric")]
        for (bench, m), g in rk.groupby(["benchmark", "metric"], sort=False):
            if len(g) < 2:
                continue
            c = g[g.implementation == "cxx"][arms].to_numpy(float).ravel()
            p = g[g.implementation == "py"][arms].to_numpy(float).ravel()
            ok = np.isfinite(c) & np.isfinite(p)
            tau = float(ss.kendalltau(c[ok], p[ok]).statistic) if ok.sum() >= 2 else float("nan")
            agree.append(dict(benchmark=bench, metric=m, kendall_tau_of_medians=tau,
                              best_cxx=np.array(arms)[ok][np.argmin(c[ok])], best_py=np.array(arms)[ok][np.argmin(p[ok])]))
    return omni, pd.DataFrame(pair), pd.DataFrame(agree)


# ---------------------------------------------------------------------------------- python ext.

def lasso_tables(runs: pd.DataFrame, an: dict, rng: np.random.Generator):
    ext = runs[runs.study == "pyext"]
    if ext.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summ, comp = [], []
    for bench, g in ext.groupby("benchmark", sort=False):
        arms = list(dict.fromkeys(g["arm"]))
        med = {}
        for arm in arms:
            h = g[g.arm == arm]
            rec = dict(benchmark=bench, arm=arm, n=len(h))
            for m in EXT_METRICS:
                v = h[m].to_numpy(float)
                rec[f"median_{m}"] = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
                rec[f"iqr_{m}"] = S.iqr(v[np.isfinite(v)]) if np.isfinite(v).sum() > 1 else float("nan")
            med[arm] = (rec["median_grid_mae_eps"], rec["median_n_params"])
            summ.append(rec)
        # Pareto front on (median error, median #coefficients), both minimised
        for rec in summ:
            if rec["benchmark"] != bench:
                continue
            e, k = med[rec["arm"]]
            rec["pareto_optimal"] = not any((e2 <= e and k2 <= k) and (e2 < e or k2 < k) for a2, (e2, k2) in med.items()
                                            if a2 != rec["arm"])
        lassos = [a for a in arms if a.startswith("lasso")]
        bases = [a for a in arms if not a.startswith("lasso")]
        for m in ("grid_mae_eps", "n_params"):
            recs, ps = [], []
            wide = g.pivot_table(index="run_id", columns="arm", values=m)
            for la in lassos:
                for ba in bases:
                    d = (wide[la] - wide[ba]).dropna().to_numpy()
                    if len(d) < 2:
                        continue
                    sh = S.hl_paired(d)
                    ci = S.bootstrap_ci(S.hl_paired, [d], an["n_bootstrap"], rng, paired=True)
                    margin = _margin(m, an, wide[ba].dropna().to_numpy())
                    recs.append(dict(benchmark=bench, metric=m, lasso=la, baseline=ba, n_pairs=len(d),
                                     median_lasso=float(wide[la].median()), median_baseline=float(wide[ba].median()),
                                     hl_shift_lasso_minus_base=sh, ci95_lo=ci[0.95][0], ci95_hi=ci[0.95][1],
                                     ci90_lo=ci[0.90][0], ci90_hi=ci[0.90][1],
                                     a12_lasso_gt_base=S.a12(wide[la].dropna(), wide[ba].dropna()),
                                     p_wilcoxon=S.wilcoxon_p(d), margin=margin))
                    ps.append(recs[-1]["p_wilcoxon"])
            for r, pa in zip(recs, S.holm(ps)):
                r["p_holm"] = pa
                r["verdict"] = S.verdict(pa, (r["ci90_lo"], r["ci90_hi"]), (r["ci95_lo"], r["ci95_hi"]),
                                         r["hl_shift_lasso_minus_base"], r["margin"], an["alpha"])
                comp.append(r)
    comp = pd.DataFrame(comp)
    tradeoff = []
    if not comp.empty:
        for (bench, la, ba), g in comp.groupby(["benchmark", "lasso", "baseline"], sort=False):
            e = g[g.metric == "grid_mae_eps"].iloc[0]
            k = g[g.metric == "n_params"].iloc[0]
            e_better, e_worse = e.p_holm < an["alpha"] and e.hl_shift_lasso_minus_base < 0, e.p_holm < an["alpha"] and e.hl_shift_lasso_minus_base > 0
            k_better, k_worse = k.p_holm < an["alpha"] and k.hl_shift_lasso_minus_base < 0, k.p_holm < an["alpha"] and k.hl_shift_lasso_minus_base > 0
            e_equiv = e.verdict in ("equivalent", "different but negligible")
            if (e_better or e_equiv) and k_better:
                cls = "more compact, accuracy not worse" if not e_better else "dominates (more accurate and more compact)"
            elif e_better and not k_worse:
                cls = "more accurate, not larger"
            elif e_worse and k_better:
                cls = "trade-off: more compact but less accurate"
            elif e_better and k_worse:
                cls = "trade-off: more accurate but larger"
            elif e_worse or k_worse:
                cls = "dominated (worse, no compensating gain)"
            else:
                cls = "no detectable difference"
            tradeoff.append(dict(benchmark=bench, lasso=la, baseline=ba,
                                 d_err_eps=e.hl_shift_lasso_minus_base, err_verdict=e.verdict,
                                 d_params=k.hl_shift_lasso_minus_base, params_verdict=k.verdict, classification=cls))
    return pd.DataFrame(summ), comp, pd.DataFrame(tradeoff)


# -------------------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_parity_curves(curves: pd.DataFrame, runs: pd.DataFrame, out: Path) -> List[str]:
    plt = _plt()
    files = []
    par = curves[curves.study == "parity"]
    for bench, g in par.groupby("benchmark", sort=False):
        arms = list(dict.fromkeys(g["arm"]))
        fig, axes = plt.subplots(2, len(arms), figsize=(4.2 * len(arms), 6), sharex=True, squeeze=False)
        for j, arm in enumerate(arms):
            for impl, h in g[g.arm == arm].groupby("implementation"):
                q = h.groupby("step").agg(m=("mae_eps", "median"), lo=("mae_eps", lambda v: v.quantile(.25)),
                                          hi=("mae_eps", lambda v: v.quantile(.75)), k=("macro", "median"),
                                          klo=("macro", lambda v: v.quantile(.25)), khi=("macro", lambda v: v.quantile(.75)))
                c = IMPL_COLOR[impl]
                axes[0, j].plot(q.index, q.m, color=c, lw=1.2, label=IMPL_LABEL[impl])
                axes[0, j].fill_between(q.index, q.lo, q.hi, color=c, alpha=.2, lw=0)
                axes[1, j].plot(q.index, q.k, color=c, lw=1.2)
                axes[1, j].fill_between(q.index, q.klo, q.khi, color=c, alpha=.2, lw=0)
            axes[0, j].axhline(1.0, color="k", lw=.8, ls="--")
            axes[0, j].set_yscale("log")
            axes[0, j].set_title(arm)
            axes[1, j].set_xlabel("learning steps")
            for ax in axes[:, j]:
                ax.grid(alpha=.3)
        axes[0, 0].set_ylabel("online test MAE / ε0 (median, IQR)")
        axes[1, 0].set_ylabel("macroclassifiers")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Parity — {bench}", fontsize=11)
        fig.tight_layout()
        f = out / f"parity_curves_{bench}.png"
        fig.savefig(f, dpi=120)
        plt.close(fig)
        files.append(f.name)
    return files


def fig_parity_boxes(runs: pd.DataFrame, out: Path) -> List[str]:
    plt = _plt()
    par = runs[runs.study == "parity"]
    if par.empty:
        return []
    benches = list(dict.fromkeys(par.benchmark))
    files = []
    for metric, label, log in (("grid_mae_eps", "grid MAE / ε0", True), ("n_macro", "macroclassifiers", False)):
        fig, axes = plt.subplots(1, len(benches), figsize=(3.6 * len(benches), 3.8), squeeze=False)
        for ax, bench in zip(axes[0], benches):
            g = par[par.benchmark == bench]
            arms = list(dict.fromkeys(g.arm))
            data, pos, cols = [], [], []
            for i, arm in enumerate(arms):
                for k, impl in enumerate(("cxx", "py")):
                    v = g[(g.arm == arm) & (g.implementation == impl)][metric].dropna().to_numpy()
                    if len(v):
                        data.append(v); pos.append(i * 3 + k); cols.append(IMPL_COLOR[impl])
            bp = ax.boxplot(data, positions=pos, widths=.8, patch_artist=True, showfliers=True)
            for patch, c in zip(bp["boxes"], cols):
                patch.set_facecolor(c); patch.set_alpha(.45)
            ax.set_xticks([i * 3 + .5 for i in range(len(arms))])
            ax.set_xticklabels(arms, fontsize=8)
            if log:
                ax.set_yscale("log")
            ax.set_title(bench, fontsize=9)
            ax.grid(alpha=.3)
        axes[0, 0].set_ylabel(label)
        handles = [plt.Rectangle((0, 0), 1, 1, color=IMPL_COLOR[i], alpha=.45) for i in ("cxx", "py")]
        axes[0, -1].legend(handles, [IMPL_LABEL["cxx"], IMPL_LABEL["py"]], fontsize=8)
        fig.tight_layout()
        f = out / f"parity_box_{metric}.png"
        fig.savefig(f, dpi=120)
        plt.close(fig)
        files.append(f.name)
    return files


def fig_profiles(profiles: pd.DataFrame, out: Path) -> List[str]:
    plt = _plt()
    files = []
    if profiles is None or profiles.empty:
        return files
    for (study, bench), g in profiles.groupby(["study", "benchmark"], sort=False):
        arms = list(dict.fromkeys(g.arm))
        fig, axes = plt.subplots(len(arms), 1, figsize=(9, 2.1 * len(arms)), sharex=True, squeeze=False)
        for ax, arm in zip(axes[:, 0], arms):
            h = g[g.arm == arm]
            first = h[h.implementation == h.implementation.iloc[0]]
            ax.plot(first.x, first.y_true, color="k", lw=1, label="target")
            for impl, q in h.groupby("implementation"):
                ax.plot(q.x, q.pred_median, color=IMPL_COLOR.get(impl, "C2"), lw=1, label=f"{IMPL_LABEL.get(impl, impl)} (median)")
                ax.fill_between(q.x, q.pred_q25, q.pred_q75, color=IMPL_COLOR.get(impl, "C2"), alpha=.2, lw=0)
            ax.set_ylabel(arm, fontsize=8)
            ax.grid(alpha=.3)
        axes[0, 0].legend(fontsize=7, ncol=3)
        axes[-1, 0].set_xlabel("x")
        fig.suptitle(f"{study} — {bench}: predictions on the evaluation grid", fontsize=10)
        fig.tight_layout()
        f = out / f"{study}_predictions_{bench}.png"
        fig.savefig(f, dpi=110)
        plt.close(fig)
        files.append(f.name)
    return files


def fig_pareto(summ: pd.DataFrame, out: Path) -> List[str]:
    if summ.empty:
        return []
    plt = _plt()
    benches = list(dict.fromkeys(summ.benchmark))
    fig, axes = plt.subplots(1, len(benches), figsize=(4.4 * len(benches), 4), squeeze=False)
    for ax, bench in zip(axes[0], benches):
        g = summ[summ.benchmark == bench]
        for r in g.itertuples():
            lasso = r.arm.startswith("lasso")
            ax.errorbar(r.median_n_params, r.median_grid_mae_eps, xerr=r.iqr_n_params / 2, yerr=r.iqr_grid_mae_eps / 2,
                        fmt="o" if lasso else "s", color="C3" if lasso else "C0", ms=5, alpha=.85,
                        mec="k" if r.pareto_optimal else "none")
            ax.annotate(r.arm, (r.median_n_params, r.median_grid_mae_eps), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("non-zero coefficients (median)")
        ax.set_yscale("log")
        ax.set_title(bench, fontsize=9)
        ax.grid(alpha=.3)
    axes[0, 0].set_ylabel("grid MAE / ε0 (median)")
    fig.suptitle("Python-only extension: accuracy vs compactness (black edge = Pareto-optimal)", fontsize=10)
    fig.tight_layout()
    f = out / "pyext_pareto.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    return [f.name]


def fig_checkpoints(cks: pd.DataFrame, out: Path) -> List[str]:
    if cks is None or cks.empty:
        return []
    plt = _plt()
    files = []
    for study, gs in cks.groupby("study"):
        benches = list(dict.fromkeys(gs.benchmark))
        fig, axes = plt.subplots(1, len(benches), figsize=(4.4 * len(benches), 3.8), squeeze=False)
        for ax, bench in zip(axes[0], benches):
            g = gs[gs.benchmark == bench]
            for i, (arm, h) in enumerate(g.groupby("arm", sort=False)):
                q = h.groupby("step")["grid_mae_eps"].median()
                ax.plot(q.index, q.values, marker=".", lw=1, label=arm, color=f"C{i % 10}")
            ax.axhline(1, color="k", ls="--", lw=.8)
            ax.set_yscale("log")
            ax.set_title(bench, fontsize=9)
            ax.set_xlabel("learning steps")
            ax.grid(alpha=.3)
        axes[0, 0].set_ylabel("grid MAE / ε0 (median)")
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(f"{study}: generalisation error on the grid during training (Python only)", fontsize=10)
        fig.tight_layout()
        f = out / f"{study}_grid_checkpoints.png"
        fig.savefig(f, dpi=120)
        plt.close(fig)
        files.append(f.name)
    return files


# --------------------------------------------------------------------------------------- report

def analyze(cdir: Path):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # degenerate samples (e.g. zero variance) in scipy tests
    P = campaign_paths(cdir)
    D = P["derived"]
    if not (D / "runs.csv").exists():
        raise SystemExit("[fail] derived/runs.csv missing: run collect first")
    cfg = json.loads((P["manifest"] / "config.resolved.json").read_text())
    an = cfg["analysis"]
    rng = np.random.default_rng(an["bootstrap_seed"])
    runs = pd.read_csv(D / "runs.csv")
    curves = pd.read_csv(D / "curves.csv.gz")
    cks = pd.read_csv(D / "checkpoints.csv.gz") if (D / "checkpoints.csv.gz").exists() else pd.DataFrame()
    profiles = pd.read_csv(D / "grid_profiles.csv.gz") if (D / "grid_profiles.csv.gz").exists() else pd.DataFrame()
    T = D / "tables"
    F = D / "figures"
    T.mkdir(exist_ok=True)
    F.mkdir(exist_ok=True)

    par = parity_tables(runs, an, rng)
    omni, pair, agree = predictor_tables(runs, an)
    summ, comp, trade = lasso_tables(runs, an, rng)
    for name, df in (("parity_tests", par), ("predictors_omnibus", omni), ("predictors_pairwise", pair),
                     ("predictors_rank_agreement", agree), ("pyext_summary", summ), ("pyext_vs_baselines", comp),
                     ("pyext_tradeoffs", trade)):
        if df is not None and not df.empty:
            df.to_csv(T / f"{name}.csv", index=False)
    desc = runs.groupby(["study", "implementation", "benchmark", "arm"], sort=False).agg(
        n=("run_id", "count"), grid_mae_eps_median=("grid_mae_eps", "median"),
        grid_mae_eps_iqr=("grid_mae_eps", S.iqr), n_macro_median=("n_macro", "median"),
        n_params_median=("n_params", "median"), converged_frac=("converged", "mean"),
        steps_to_eps_median=("steps_to_eps", "median"), runtime_s_median=("runtime_s", "median"),
        degenerate_rules_median=("n_degenerate_rules", "median")).reset_index()
    desc.to_csv(T / "descriptives.csv", index=False)

    figs = []
    figs += fig_parity_curves(curves, runs, F)
    figs += fig_parity_boxes(runs, F)
    figs += fig_profiles(profiles, F)
    figs += fig_pareto(summ, F)
    figs += fig_checkpoints(cks, F)

    write_report(cdir, cfg, runs, desc, par, omni, pair, agree, summ, comp, trade, figs)
    print(f"[ok] analysis -> {D}")


def write_report(cdir, cfg, runs, desc, par, omni, pair, agree, summ, comp, trade, figs):
    P = campaign_paths(cdir)
    an = cfg["analysis"]
    L = []
    w = L.append
    w(f"# XCSF Python vs C++ — campaign report `{cfg['campaign_id']}`\n")
    w(f"_Generated from `raw/` by `python -m xcsfcamp analyze`. Profile: **{cfg.get('profile', 'full')}**. "
      "Every number below is regenerated from the immutable raw results._\n")
    if cfg.get("profile") == "smoke":
        w("> **Smoke profile**: tiny budget and 2 runs — this report only proves the pipeline works; "
          "its statistics are meaningless.\n")
    x = cfg["xcsf"]
    w("## Setup\n")
    w(f"- runs per cell: **{cfg['n_runs']}** (seeds {cfg['seed_base']}…{cfg['seed_base'] + cfg['n_runs'] - 1}, shared by both implementations)")
    w(f"- learning problems N = {x['n_learning_problems']}, population size = {x['population_size']}, "
      f"epsilon0 = {cfg['epsilon_fraction']} × output range, r0 = m0 = {x['r0_fraction']} × domain width")
    w(f"- online test curve: rolling window of {cfg['monitoring']['window']} test problems (xcslib) / pre-update errors (Python)")
    w(f"- equivalence margins: ±{an['equivalence_margin_error_eps_fraction']} ε0 for error metrics, "
      f"±{an['equivalence_margin_size_relative']:.0%} of the C++ median for size metrics, "
      f"±{an.get('equivalence_margin_fraction_abs', 0.05)} for fractions; α = {an['alpha']} with Holm correction")
    w("- parameter parity table: `manifest/parity_audit.md`; semantic differences that cannot be aligned without "
      "modifying the libraries: `campaign/docs/PARAMETER_PARITY.md`\n")
    w("### Descriptive summary (medians)\n")
    w(md_table(desc))
    w("")

    if not par.empty:
        w("## Q1 / Q5 — Same configuration, same predictor: comparable approximation quality?\n")
        w("Primary metric: MAE on the shared evaluation grid divided by ε0. Shift = Hodges–Lehmann estimate of "
          "(Python − C++), bootstrap 95% CI. Verdict combines the Holm-adjusted Mann–Whitney test with an equivalence "
          "test (90% CI inside the margin).\n")
        for m in PRIMARY:
            g = par[par.metric == m][["benchmark", "predictor", "median_cxx", "iqr_cxx", "median_py", "iqr_py",
                                      "hl_shift_py_minus_cxx", "ci95_lo", "ci95_hi", "margin", "p_holm",
                                      "a12_py_gt_cxx", "effect", "verdict"]]
            w(f"### {m} — {METRICS[m][1]}\n")
            w(md_table(g))
            w("")
        w("## Q2 — Systematic differences (convergence, final error, size, stability)\n")
        sec = par[~par.metric.isin(PRIMARY)]
        cnt = par.groupby(["metric", "verdict"]).size().unstack(fill_value=0)
        w("Verdict counts over all benchmark × predictor cells:\n")
        w(md_table(cnt.reset_index()))
        w("")
        sign = par.assign(py_lower=par.hl_shift_py_minus_cxx < 0).groupby("metric").agg(
            cells=("py_lower", "size"), python_lower_in=("py_lower", "sum")).reset_index()
        w("Direction of the shift (cells where Python's median is lower; for error metrics lower is better):\n")
        w(md_table(sign))
        w("")
        w("Secondary metrics:\n")
        w(md_table(sec[["metric", "benchmark", "predictor", "median_cxx", "median_py", "hl_shift_py_minus_cxx",
                        "ci95_lo", "ci95_hi", "p_holm", "verdict"]]))
        w("")
        stab = par[par.metric.isin(PRIMARY)][["metric", "benchmark", "predictor", "iqr_cxx", "iqr_py", "iqr_ratio_py_cxx", "p_bf_holm"]]
        w("Between-run stability (dispersion across seeds; Brown–Forsythe test, Holm-adjusted):\n")
        w(md_table(stab))
        w("")
    if omni is not None and not omni.empty:
        w("## Q3 — Constant vs NLMS vs RLS (same protocol, within each implementation)\n")
        w("Python runs share the training stream across predictors (paired, Friedman/Wilcoxon); xcslib draws inputs "
          "from its own RNG (independent, Kruskal–Wallis/Mann–Whitney). Lower is better for all metrics.\n")
        w(md_table(omni.drop(columns=[c for c in omni.columns if c.startswith("median_")])))
        w("")
        if not agree.empty:
            w("Do both implementations rank the predictors the same way?\n")
            w(md_table(agree))
            w("")
        w("Pairwise comparisons (Holm within implementation × benchmark × metric): `tables/predictors_pairwise.csv`.\n")
    if summ is not None and not summ.empty:
        w("## Q4 — Python-only extension: Lasso Batch / Lasso Online\n")
        w("Separate study (`pyext`), domain-scaled inputs, never pooled with the parity results. Baselines are re-run "
          "under exactly the same protocol and seeds. `rls_standard` is the textbook RLS (V0 = δI) available only in Python.\n")
        cols = ["benchmark", "arm", "n", "median_grid_mae_eps", "median_n_params", "median_n_macro",
                "median_frac_zero_slopes", "median_runtime_s", "pareto_optimal"]
        w(md_table(summ[cols]))
        w("")
        if trade is not None and not trade.empty:
            w("Trade-off classification of each Lasso variant against each baseline (paired Wilcoxon, Holm, "
              "equivalence margins as above):\n")
            w(md_table(trade))
            w("")
    w("## Figures\n")
    for f in figs:
        w(f"- `figures/{f}`")
    w("\n## Reading guide and caveats\n")
    w("- *equivalent*: the 90% CI of the shift lies inside the practical margin (TOST at 5%).\n"
      "- *different but negligible*: statistically detectable, yet within the margin — not practically important.\n"
      "- *practically different*: significant and the 95% CI lies entirely beyond the margin.\n"
      "- *inconclusive*: neither difference nor equivalence shown — more runs needed.\n"
      "- Online curves compare the same estimator (current-model |error| on a fresh uniform point, covering allowed, "
      "no update), but on different random streams; grid metrics use bit-identical evaluation points.\n"
      "- Runtime is reported, not tested: interpreted Python vs compiled C++.\n"
      "- Known implementation differences that the campaign measures rather than removes are listed in "
      "`campaign/docs/PARAMETER_PARITY.md` (e.g. the xcslib mutation operator overwrites the upper bound with the lower "
      "one; `n_degenerate_rules` in the descriptive table counts such rules in the final populations).\n")
    (P["derived"] / "report.md").write_text("\n".join(L))
