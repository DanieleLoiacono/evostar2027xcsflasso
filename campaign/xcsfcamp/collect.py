"""raw/ -> derived/ (tidy tables).  derived/ is wiped and rebuilt on every call."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import benchmarks as B
from . import cxx as C
from .metrics import curve_metrics, grid_metrics, population_metrics
from .runs import campaign_paths, load_plan, raw_dir, state

ID_COLS = ["study", "implementation", "benchmark", "predictor_name", "arm", "run_id", "seed"]


def _load_cxx(d: Path, spec: dict):
    fn = C.files_for(spec["run_id"])
    bench = B.get(spec["benchmark"])
    st = C.parse_statistics(d / fn["statistics"])
    curve = st["curve"][["step", "mae", "macro"]].copy()
    avf = C.parse_avf(d / fn["avf"], bench.dim)
    pop = C.parse_population(d / fn["population"], bench.dim, spec["predictor"]["type"])
    done = json.loads((d / "DONE.json").read_text())
    extra = dict(runtime_s=C.parse_timing_us(d / fn["timing"]) / 1e6,
                 n_coverings=done.get("validation", {}).get("n_coverings_logged"),
                 xcslib_solution_mae_eps=st["solution"]["mae"] / spec["xcsf"]["epsilon0"],
                 unmatched_grid_frac=float("nan"), py_warnings=float("nan"))
    ex = C.parse_exec_stats(d / fn["execution_statistics"])
    extra.update(ga_runs=ex.get("GA Activations"), subsumptions=ex.get("Subsumptions"))
    return curve, avf["pred"].to_numpy(float), pop, extra


def _load_py(d: Path, spec: dict):
    res = json.loads((d / "result.json").read_text())
    curve = pd.read_csv(d / "curve.csv")[["step", "mae", "macro"]]
    grid = pd.read_csv(d / "grid_predictions.csv.gz")
    pop = pd.read_csv(d / "population.csv.gz")
    extra = dict(runtime_s=res["train_time_s"], n_coverings=res["stats"]["coverings"],
                 xcslib_solution_mae_eps=float("nan"), unmatched_grid_frac=res["unmatched_grid_frac"],
                 py_warnings=res["warnings_count"], ga_runs=res["stats"]["ga_runs"],
                 subsumptions=res["stats"]["subsumptions"])
    ck = pd.read_csv(d / "checkpoints.csv") if (d / "checkpoints.csv").stat().st_size > 1 else pd.DataFrame()
    return curve, grid["pred"].to_numpy(float), pop, extra, ck


def collect(cdir: Path, allow_incomplete: bool = False):
    P = campaign_paths(cdir)
    cfg = json.loads((P["manifest"] / "config.resolved.json").read_text())
    an = cfg["analysis"]
    plan = load_plan(cdir)
    missing = [k for k, rec in plan.items() if state(cdir, rec) != "done"]
    if missing and not allow_incomplete:
        raise SystemExit(f"[fail] {len(missing)} run(s) not published (e.g. {missing[0]}); "
                         "finish them or pass --allow-incomplete")
    if P["derived"].exists():
        shutil.rmtree(P["derived"])
    P["derived"].mkdir(parents=True)

    rows, curves, cks, profiles = [], [], [], []
    grids_cache = {}
    pred_store = {}
    for key, rec in sorted(plan.items()):
        if key in missing:
            continue
        spec = rec["spec"]
        bench = B.get(spec["benchmark"])
        d = raw_dir(cdir, key)
        if spec["implementation"] == "cxx":
            curve, pred, pop, extra = _load_cxx(d, spec)
            ck = pd.DataFrame()
        else:
            curve, pred, pop, extra, ck = _load_py(d, spec)
        if bench.name not in grids_cache:
            G = bench.grid()
            grids_cache[bench.name] = (G, bench(G))
        G, yG = grids_cache[bench.name]
        if len(pred) != len(yG):
            raise SystemExit(f"[fail] {key}: {len(pred)} grid predictions, expected {len(yG)}")
        eps0 = spec["xcsf"]["epsilon0"]
        lo, hi = bench.output_range()
        ids = {c: spec[c] for c in ID_COLS}
        row = dict(ids, epsilon0=eps0, n_learning=spec["xcsf"]["n_learning_problems"],
                   population_size=spec["xcsf"]["population_size"], input_representation=spec["input_representation"],
                   predictor_type=spec["predictor"]["type"])
        row.update(grid_metrics(pred, yG, eps0, hi - lo))
        row.update(curve_metrics(curve, eps0, spec["xcsf"]["n_learning_problems"], an["last_fraction"],
                                 an["convergence_consecutive_windows"]))
        row.update(population_metrics(pop, bench, spec["predictor"]["type"]))
        row.update(extra)
        rows.append(row)
        c = curve.copy()
        c["mae_eps"] = c["mae"] / eps0
        for k, v in ids.items():
            c[k] = v
        curves.append(c)
        if not ck.empty:
            ck = ck.copy()
            ck["grid_mae_eps"] = ck["grid_mae"] / eps0
            for k, v in ids.items():
                ck[k] = v
            cks.append(ck)
        if bench.dim == 1:
            pred_store.setdefault((spec["study"], spec["implementation"], spec["benchmark"], spec["arm"]), []).append(pred)

    runs = pd.DataFrame(rows)
    runs.to_csv(P["derived"] / "runs.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(P["derived"] / "curves.csv.gz", index=False)
    if cks:
        pd.concat(cks, ignore_index=True).to_csv(P["derived"] / "checkpoints.csv.gz", index=False)
    for (study, impl, bname, arm), preds in pred_store.items():
        G, yG = grids_cache[bname]
        A = np.vstack(preds)
        profiles.append(pd.DataFrame(dict(study=study, implementation=impl, benchmark=bname, arm=arm, x=G[:, 0], y_true=yG,
                                          pred_median=np.median(A, 0), pred_q25=np.quantile(A, .25, 0),
                                          pred_q75=np.quantile(A, .75, 0),
                                          abs_err_median=np.median(np.abs(A - yG), 0))))
    if profiles:
        pd.concat(profiles, ignore_index=True).to_csv(P["derived"] / "grid_profiles.csv.gz", index=False)
    print(f"[ok] collected {len(runs)} runs -> {P['derived'] / 'runs.csv'}")
    return runs
