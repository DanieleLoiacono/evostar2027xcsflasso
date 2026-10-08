"""xcslib (C++) side: confsys generation and parsing of xcslib's standard output files.

xcslib is run through its own experiment manager (experiment_mgr2): one
`confsys.xcsf` per run directory, `xcsf-rf -f xcsf`, and the standard result files
it writes.  Nothing here drives the learning loop.

Standard files for run r (suffix 'xcsf'):
  statistics.xcsf-<r>.gz            rolling-window lines "<exp> <steps> <avg steps> <avg target> <avg macro> <avg |err|> Testing"
                                    plus one "Solution" line (grid evaluation inside xcslib)
  avf.xcsf-<r>.gz                   "State|#" then "<x1 x2 ...>|<prediction>" on the evaluation grid
  population.xcsf-<r>.gz            final population (one macroclassifier per line)
  execution_statistics.xcsf-<r>     GA / subsumption counters
  timing-report.xcsf-<r>            elapsed time (microseconds)
"""

from __future__ import annotations

import gzip
import io
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import benchmarks as B
from .parity import verify_cxx_print

SUFFIX = "xcsf"


class CxxOutputError(RuntimeError):
    pass


def files_for(run_id: int) -> Dict[str, str]:
    r = f"{run_id:04d}"
    return dict(
        statistics=f"statistics.{SUFFIX}-{r}.gz",
        avf=f"avf.{SUFFIX}-{r}.gz",
        population=f"population.{SUFFIX}-{r}.gz",
        execution_statistics=f"execution_statistics.{SUFFIX}-{r}",
        timing=f"timing-report.{SUFFIX}-{r}",
        stdout="stdout.log",
        stderr="stderr.log",
        params_print="cxx_params_print.txt",
        confsys=f"confsys.{SUFFIX}",
        exit_code="exit_code.txt",
    )


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


def parse_statistics(path: Path) -> Dict[str, Any]:
    rows, solution = [], None
    for line in _read_text(path).splitlines():
        if not line.strip():
            continue
        f = line.split("\t")
        label = f[-1].strip()
        if label == "Testing":
            if len(f) != 7:
                raise CxxOutputError(f"{path.name}: unexpected Testing line '{line}' (rolling window statistics expected)")
            rows.append(dict(step=int(f[1]), mean_target=float(f[3]), macro=float(f[4]), mae=float(f[5])))
        elif label == "Solution":
            solution = dict(mean_target=float(f[3]), macro=float(f[4]), mae=float(f[5]))
        else:
            raise CxxOutputError(f"{path.name}: unexpected line '{line}'")
    curve = pd.DataFrame(rows)
    return dict(curve=curve, solution=solution)


def parse_avf(path: Path, dim: int) -> pd.DataFrame:
    lines = _read_text(path).splitlines()
    if not lines or not lines[0].startswith("State|"):
        raise CxxOutputError(f"{path.name}: missing 'State|' header")
    xs, ps = [], []
    for line in lines[1:]:
        if not line.strip():
            continue
        state, pred = line.split("|")
        x = [float(v) for v in state.split()]
        if len(x) != dim:
            raise CxxOutputError(f"{path.name}: state with {len(x)} inputs, expected {dim}")
        xs.append(x)
        ps.append(float(pred))
    X = np.array(xs)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(dim)])
    df["pred"] = ps
    return df


_NUM_PF = re.compile(r"^(\d+)(-?\d\.\d+e[+-]\d+|-?nan|-?inf)$")
_NUM_VEC = re.compile(r"^(\d+)\[(.*)\]$")


def parse_population(path: Path, dim: int, predictor_type: str) -> pd.DataFrame:
    """Parse a xcslib XCSF population dump (with __NICHE_TRACKING__ fields).

    Columns: id, condition, action, prediction, error, fitness, set_size, experience,
    numerosity+prediction-function, creation, timestamp, as_timestamp, [as list].
    xcslib writes the prediction function right after the numerosity without a
    separator: '40[w0;w1]' (nlms/rls) or '309.11862e+01' (value: numerosity 30,
    value 9.11862e+01, scientific with one leading digit).
    """
    recs = []
    for line in _read_text(path).splitlines():
        if not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 9:
            raise CxxOutputError(f"{path.name}: malformed classifier line '{line[:120]}'")
        intervals = re.findall(r"\[([^;\]]+);([^\]]+)\]", f[1])
        if len(intervals) != dim:
            raise CxxOutputError(f"{path.name}: condition '{f[1]}' has {len(intervals)} intervals, expected {dim}")
        tail = f[8]
        if predictor_type == "constant":
            m = _NUM_PF.match(tail)
            if not m:
                raise CxxOutputError(f"{path.name}: cannot split numerosity/value in '{tail}'")
            num, weights = int(m.group(1)), [float(m.group(2))]
        else:
            m = _NUM_VEC.match(tail)
            if not m:
                raise CxxOutputError(f"{path.name}: cannot split numerosity/weights in '{tail}'")
            num, weights = int(m.group(1)), [float(w) for w in m.group(2).split(";")]
        rec = dict(id=int(f[0]), numerosity=num, error=float(f[4]), fitness=float(f[5]), set_size=float(f[6]),
                   experience=int(float(f[7])), weights=weights)
        for i, (lo, hi) in enumerate(intervals):
            rec[f"lower{i}"] = float(lo)
            rec[f"upper{i}"] = float(hi)
        recs.append(rec)
    return pd.DataFrame(recs)


def parse_timing_us(path: Path) -> float:
    m = re.findall(r"Total Elapsed Time = (\d+)", path.read_text())
    if not m:
        raise CxxOutputError(f"{path.name}: no 'Total Elapsed Time'")
    return float(m[-1])


def parse_exec_stats(path: Path) -> Dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"#\s*(.+?)\s+(-?\d+)\s*$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def count_coverings(stderr_path: Path) -> int:
    n = 0
    opener = gzip.open if stderr_path.suffix == ".gz" else open
    with opener(stderr_path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("COVERING"):
                n += 1
    return n


def validate_run_dir(run_dir: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Fail loudly unless the run directory holds a complete, sane xcslib run."""
    fn = files_for(spec["run_id"])
    problems: List[str] = []
    for k in ("confsys", "params_print", "exit_code", "statistics", "avf", "population", "execution_statistics", "timing"):
        p = run_dir / fn[k]
        if not p.exists() or p.stat().st_size == 0:
            problems.append(f"missing or empty output: {fn[k]}")
    if problems:
        raise CxxOutputError("; ".join(problems))
    rc = (run_dir / fn["exit_code"]).read_text().strip()
    if rc != "0":
        raise CxxOutputError(f"xcsf-rf exited with code {rc}")

    mism = verify_cxx_print(spec, (run_dir / fn["params_print"]).read_text())
    if mism:
        raise CxxOutputError("xcslib parsed parameters differ from the intended ones:\n  " + "\n  ".join(mism))

    N, W = spec["xcsf"]["n_learning_problems"], spec["monitoring"]["window"]
    st = parse_statistics(run_dir / fn["statistics"])
    curve = st["curve"]
    if len(curve) != N // W:
        raise CxxOutputError(f"statistics: {len(curve)} Testing lines, expected {N // W} (N={N}, window={W})")
    if int(curve["step"].iloc[-1]) != N:
        raise CxxOutputError(f"statistics: last step {curve['step'].iloc[-1]} != N={N}")
    if not np.all(np.isfinite(curve[["mae", "macro", "mean_target"]].to_numpy())):
        raise CxxOutputError("statistics: NaN/inf in rolling statistics")
    if st["solution"] is None:
        raise CxxOutputError("statistics: no 'Solution' line (evaluate solution must be on)")

    bench = B.get(spec["benchmark"])
    avf = parse_avf(run_dir / fn["avf"], bench.dim)
    grid = bench.grid()
    if len(avf) != len(grid):
        raise CxxOutputError(f"avf: {len(avf)} grid points, expected {len(grid)}")
    X = avf[[f"x{i}" for i in range(bench.dim)]].to_numpy()
    # xcslib prints states with 6 significant digits
    tol = 5.0001e-6 * np.maximum(1.0, np.abs(grid)) * 10
    if np.any(np.abs(X - grid) > tol):
        i = int(np.argmax(np.max(np.abs(X - grid) - tol, axis=1)))
        raise CxxOutputError(f"avf: grid point {i} = {X[i]} differs from the campaign grid {grid[i]}")
    if not np.all(np.isfinite(avf["pred"].to_numpy())):
        raise CxxOutputError("avf: NaN/inf predictions")

    pop = parse_population(run_dir / fn["population"], bench.dim, spec["predictor"]["type"])
    if pop.empty:
        raise CxxOutputError("population: empty")
    total = int(pop["numerosity"].sum())
    if total > spec["xcsf"]["population_size"] + 2 or total <= 0:
        raise CxxOutputError(f"population: sum of numerosities {total} exceeds N={spec['xcsf']['population_size']}")
    W_all = np.concatenate([np.asarray(w, float) for w in pop["weights"]])
    if not np.all(np.isfinite(W_all)) or not np.all(np.isfinite(pop[["error", "fitness"]].to_numpy())):
        raise CxxOutputError("population: NaN/inf in classifier parameters")
    lo, hi = bench.output_range()
    if spec["predictor"]["type"] == "constant" and np.max(np.abs(W_all)) > 1e3 * max(abs(lo), abs(hi), 1.0):
        raise CxxOutputError("population: absurd constant prediction (value_pf::prediction is uninitialised in xcslib)")
    timing_us = parse_timing_us(run_dir / fn["timing"])
    return dict(n_curve=len(curve), n_grid=len(avf), n_macro=len(pop), n_micro=total, timing_us=timing_us,
                solution_mae_xcslib=st["solution"]["mae"])
