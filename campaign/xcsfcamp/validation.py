"""Pre-campaign validation (run before planning the full campaign).

1. Benchmark unit tests: hand-calculated points, float inputs preserved, finite
   outputs over the domain, evaluation-grid endpoints.
2. Reference plots of every target function (saved before any XCSF run).
3. Predictor-level parity: xcsf_python LocalPredictor (with the campaign mapping)
   vs a NumPy transcription of the xcslib update rules (value.cpp, nlms.cpp, rls.cpp,
   rls_delta.cpp) on identical sample sequences, including offspring (clone) semantics.
3b. C++ predictor check (requires the build): the compiled xcslib prediction functions,
   driven sample by sample by campaign/build/bin/pf_driver, vs the same transcriptions.
4. C++ benchmark parity: xcslib's own environment (execution trace) vs the
   Python definitions, for every benchmark (requires the built binary).
5. Determinism: identical seeds reproduce identical results (both implementations).
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import benchmarks as B


class ValidationError(AssertionError):
    pass


def _check(cond, msg):
    if not cond:
        raise ValidationError(msg)


# ------------------------------------------------------------------------------------------- 1
def test_benchmarks(log):
    for b in B.BENCHMARKS.values():
        for x, y in b.known_points:
            got = float(b(np.array([x]))[0])
            _check(abs(got - y) <= 1e-9 * max(1, abs(y)), f"{b.name}: f({x}) = {got}, expected {y}")
        rng = np.random.default_rng(0)
        X = b.sample(rng, 10000)
        _check(X.dtype == np.float64, f"{b.name}: samples are not float64")
        _check(np.all((X >= np.array(b.lower)) & (X < np.array(b.upper))), f"{b.name}: samples outside domain")
        frac_non_int = float(np.mean(X != np.round(X)))
        _check(frac_non_int > 0.999, f"{b.name}: inputs look discretised ({frac_non_int:.4f} non-integer)")
        Xf = X.copy()
        _check(np.array_equal(b(X), b(Xf)) and np.array_equal(X, Xf), f"{b.name}: function mutates/rounds inputs")
        _check(np.all(np.isfinite(b(X))), f"{b.name}: non-finite outputs")
        lo, hi = b.output_range()
        yy = b(X)
        _check(yy.min() >= lo - 1e-9 and yy.max() <= hi + 1e-9, f"{b.name}: outputs outside declared range")
        G = b.grid()
        _check(np.allclose(G[0], b.lower, atol=0, rtol=0), f"{b.name}: grid does not start at the lower corner")
        _check(np.all(G <= np.array(b.upper)), f"{b.name}: grid exceeds the domain")
        _check(np.all(np.array(b.upper) - G[-1] < b.grid_resolution), f"{b.name}: grid last point too far from upper corner")
        _check(np.all(np.isfinite(b(G))), f"{b.name}: non-finite outputs on the grid")
        if b.dim == 1:
            ok_end = abs(G[-1, 0] - b.upper[0]) < 1e-9
            log(f"  {b.name:20s} grid {len(G)} pts [{G[0,0]:g}, {G[-1,0]:.12g}]{'' if ok_end else ' (upper end excluded by float accumulation, as in xcslib)'}"
                f"  range [{lo:.6g}, {hi:.6g}]  eps0(5%)={b.epsilon0(0.05):.6g}")
        else:
            log(f"  {b.name:20s} grid {len(G)} pts  range [{lo:.6g}, {hi:.6g}]")
    log("[ok] benchmark unit tests")


# ------------------------------------------------------------------------------------------- 2
def plot_benchmarks(out: Path, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    one = [b for b in B.BENCHMARKS.values() if b.dim == 1]
    fig, axes = plt.subplots(len(one), 1, figsize=(8, 2.2 * len(one)), constrained_layout=True)
    for ax, b in zip(axes, one):
        x = np.linspace(b.lower[0], b.upper[0], 5001)[:, None]
        ax.plot(x[:, 0], b(x), lw=1.2)
        e = b.epsilon0(0.05)
        ax.set_title(f"{b.name}   f range {b.output_range()[1]-b.output_range()[0]:.4g}   eps0(5%) = {e:.4g}", fontsize=9)
        ax.grid(alpha=.3)
    fig.savefig(out / "benchmarks_1d.png", dpi=130)
    plt.close(fig)
    b = B.get("sin_cos_surface_2d")
    g = np.linspace(0, 1, 201)
    XX, YY = np.meshgrid(g, g)
    Z = b(np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(Z, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r")
    fig.colorbar(im)
    ax.set_title(b.name, fontsize=9)
    fig.savefig(out / "benchmark_sin_cos_surface_2d.png", dpi=130)
    plt.close(fig)
    log(f"[ok] reference plots -> {out}")


# ------------------------------------------------------------------------------------------- 3
def _xcslib_value(phis, ys, eta):
    w, out = 0.0, []
    for y in ys:
        w += eta * (y - w)
        out.append(w)
    return np.array(out)


def _xcslib_nlms(phis, ys, eta, x0):
    w = np.zeros(phis.shape[1])
    out = []
    for phi, y in zip(phis, ys):
        x = phi[1:]
        err = y - phi @ w
        xpow2 = x0 * x0 + np.sum(x * x)
        corr = eta * err / xpow2
        w[0] += x0 * corr
        w[1:] += corr * x
        out.append(phi @ w)
    return np.array(out)


def _xcslib_rls(phis, ys):
    p = phis.shape[1]
    w, V = np.zeros(p), np.zeros((p, p))      # rls_pf: delta never read -> V0 = 0 * I
    out = []
    for phi, y in zip(phis, ys):
        err = y - phi @ w
        num = V @ phi
        K = num / (1.0 + phi @ num)
        V = V - np.outer(K, phi) @ V + np.eye(p)  # "add Q": identity added at every update
        w = w + K * err
        out.append(phi @ w)
    return np.array(out)


def _xcslib_rls_delta(phis, ys, delta, clone_at=0):
    """rls_delta.cpp (= Lanzi et al. 2005, Alg. 5): V0 = delta*I, V <- V - (V phi)(V phi)^T / beta."""
    p = phis.shape[1]
    w, V = np.zeros(p), np.eye(p) * delta
    out = []
    for t, (phi, y) in enumerate(zip(phis, ys), 1):
        err = y - phi @ w
        num = V @ phi
        beta = 1.0 + phi @ num
        V = V - np.outer(num, num) / beta
        w = w + (num / beta) * err
        out.append(phi @ w)
        if t == clone_at:                      # offspring: weights inherited, V restarted
            V = np.eye(p) * delta
    return np.array(out)


def _xcslib_rls_clone(phis, ys, clone_at=0):
    p = phis.shape[1]
    w, V = np.zeros(p), np.zeros((p, p))
    out = []
    for t, (phi, y) in enumerate(zip(phis, ys), 1):
        err = y - phi @ w
        num = V @ phi
        K = num / (1.0 + phi @ num)
        V = V - np.outer(K, phi) @ V + np.eye(p)
        w = w + K * err
        out.append(phi @ w)
        if t == clone_at:
            V = np.zeros((p, p))
    return np.array(out)


def _run_local(pred, phis, y, clone_at=0):
    got = []
    for t, (phi, target) in enumerate(zip(phis, y), 1):
        pred.update(phi, target)
        got.append(float(pred.predict(phi)))
        if t == clone_at:
            pred = pred.offspring()
    return np.array(got)


def test_predictor_parity(log):
    import sys
    from .config import PY_LIB_DIR
    sys.path.insert(0, str(PY_LIB_DIR / "src"))
    from xcsf.prediction import LocalPredictor, design_matrix
    from .parity import PY_RLS_XCSLIB_DELTA

    rng = np.random.default_rng(42)
    for lo, hi in ((0.0, 100.0), (1000.0, 1100.0), (0.0, 1.0)):
        X = rng.uniform(lo, hi, (500, 1))
        y = 100 * np.sin(2 * np.pi * X[:, 0] / (hi - lo)) + rng.normal(0, 1, 500)
        phis = design_matrix(X, 1, 1.0)
        cases = {
            "constant": (LocalPredictor(2, method="constant", learning_rate=0.2, initial_prediction=0.0),
                         _xcslib_value(phis, y, 0.2)),
            "nlms": (LocalPredictor(2, method="nlms", learning_rate=0.2, x0=1.0), _xcslib_nlms(phis, y, 0.2, 1.0)),
            "rls_xcslib": (LocalPredictor(2, method="rlsk", delta=PY_RLS_XCSLIB_DELTA, process_noise=1.0,
                                          forgetting_factor=1.0, kalman_noise=False, x0=1.0), _xcslib_rls(phis, y)),
            "rls_delta": (LocalPredictor(2, method="rlsk", delta=1000.0, process_noise=0.0, forgetting_factor=1.0,
                                         kalman_noise=False, x0=1.0), _xcslib_rls_delta(phis, y, 1000.0)),
            # same estimator in QR (square-root information) form: Python-only arm rls_standard
            "rls_standard": (LocalPredictor(2, method="rls", delta=1000.0, forgetting_factor=1.0, x0=1.0),
                             _xcslib_rls_delta(phis, y, 1000.0)),
        }
        for name, (pred, ref) in cases.items():
            got = []
            for phi, t in zip(phis, y):
                pred.update(phi, t)
                got.append(float(pred.predict(phi)))
            got = np.array(got)
            dev = np.max(np.abs(got - ref)) / max(1.0, np.max(np.abs(ref)))
            _check(dev < 1e-6, f"predictor parity {name} on [{lo},{hi}]: max relative deviation {dev:.3g}")
            log(f"  {name:12s} x in [{lo:g},{hi:g}]  max rel. deviation from xcslib formula {dev:.2e}")
        # offspring semantics: weights inherited, covariance restarted from delta*I
        for name, pred, ref in (
            ("rls_delta", LocalPredictor(2, method="rlsk", delta=1000.0, process_noise=0.0, forgetting_factor=1.0,
                                         kalman_noise=False, x0=1.0), _xcslib_rls_delta(phis, y, 1000.0, 200)),
            ("rls_xcslib", LocalPredictor(2, method="rlsk", delta=PY_RLS_XCSLIB_DELTA, process_noise=1.0,
                                          forgetting_factor=1.0, kalman_noise=False, x0=1.0),
             _xcslib_rls_clone(phis, y, 200))):
            got = _run_local(pred, phis, y, clone_at=200)
            dev = np.max(np.abs(got - ref)) / max(1.0, np.max(np.abs(ref)))
            _check(dev < 1e-6, f"predictor parity {name} with offspring on [{lo},{hi}]: max relative deviation {dev:.3g}")
            log(f"  {name:12s} x in [{lo:g},{hi:g}]  with offspring at t=200: max rel. deviation {dev:.2e}")
    log("[ok] predictor-level parity (campaign mapping reproduces the xcslib update rules)")


def _pf_driver_confsys(ptype: str, x0: float, delta: float) -> str:
    pf = {"nlms": "nlms", "rls_xcslib": "rls", "rls_delta": "rls_delta"}[ptype]
    text = (f"<prediction::base>\n\tinput size = 1\n\tprediction function = {pf}\n\tdegree = 1\n</prediction::base>\n"
            f"<prediction::nlms>\n\tlearning rate = 0.2\n\tx0 = {x0!r}\n</prediction::nlms>\n")
    if ptype == "rls_xcslib":
        text += f"<prediction::rls>\n\tx0 = {x0!r}\n</prediction::rls>\n"
    if ptype == "rls_delta":
        text += f"<prediction::rls_delta>\n\tx0 = {x0!r}\n\tdelta = {delta!r}\n</prediction::rls_delta>\n"
    return text


def _pf_driver_run(driver: Path, ptype: str, X: np.ndarray, y: np.ndarray, x0: float, delta: float, clone_at: int):
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "confsys.pfd").write_text(_pf_driver_confsys(ptype, x0, delta))
        inp = "\n".join(f"{x!r} {t!r}" for x, t in zip(X[:, 0].tolist(), y.tolist())) + "\n"
        r = subprocess.run([str(driver), "pfd", str(clone_at)], cwd=td, input=inp, capture_output=True, text=True,
                           timeout=120)
    if r.returncode != 0:
        raise ValidationError(f"pf_driver ({ptype}) failed:\n{r.stderr[-2000:]}")
    preds = np.array([float(line.split(" ", 1)[0]) for line in r.stdout.splitlines() if line.strip()])
    _check(len(preds) == len(y), f"pf_driver ({ptype}): {len(preds)} outputs for {len(y)} samples")
    return preds, r.stderr


def test_cxx_predictors(driver: Path, log):
    """Compiled xcslib prediction functions vs the NumPy transcriptions (and hence the Python mapping)."""
    rng = np.random.default_rng(43)
    for lo, hi in ((0.0, 100.0), (1000.0, 1100.0), (1000.0, 1025.0), (0.0, 1.0)):
        X = rng.uniform(lo, hi, (500, 1))
        y = 100 * np.sin(2 * np.pi * X[:, 0] / 100.0) + rng.normal(0, 1, 500)
        phis = np.column_stack([np.ones(len(X)), X])
        for ptype, clone_at in (("nlms", 0), ("rls_xcslib", 0), ("rls_xcslib", 200), ("rls_delta", 0), ("rls_delta", 200)):
            got, err = _pf_driver_run(driver, ptype, X, y, 1.0, 1000.0, clone_at)
            ref = {"nlms": lambda: _xcslib_nlms(phis, y, 0.2, 1.0),
                   "rls_xcslib": lambda: _xcslib_rls_clone(phis, y, clone_at),
                   "rls_delta": lambda: _xcslib_rls_delta(phis, y, 1000.0, clone_at)}[ptype]()
            dev = np.max(np.abs(got - ref)) / max(1.0, np.max(np.abs(ref)))
            _check(dev < 1e-9, f"C++ predictor {ptype} (clone at {clone_at}) on [{lo},{hi}]: "
                               f"max relative deviation from the transcription {dev:.3g}")
            if ptype == "rls_delta":
                _check("*** rls_delta: x0 = 1 delta = 1000" in err, "rls_delta did not log the parsed x0/delta")
            log(f"  C++ {ptype:10s} clone@{clone_at:<3d} x in [{lo:g},{hi:g}]  max rel. deviation {dev:.2e}")
    log("[ok] C++ prediction functions (compiled xcslib code) match their transcriptions")


# ------------------------------------------------------------------------------------------- 4
def _make_spec(bench_name, ptype="nlms", n=200, seed=1001, run_id=0):
    from .config import CAMPAIGN_DIR, load_config, resolve_xcsf
    cfg = load_config(CAMPAIGN_DIR / "config" / "campaign.json")
    bench = B.get(bench_name)
    x = resolve_xcsf(cfg, bench, "raw")
    x.update(n_learning_problems=n, population_size=100)
    pred = {"constant": {"type": "constant", "eta": 0.2}, "nlms": {"type": "nlms", "eta": 0.2, "x0": 1.0},
            "rls_xcslib": {"type": "rls_xcslib", "x0": 1.0},
            "rls_delta": {"type": "rls_delta", "x0": 1.0, "delta": 1000.0}}[ptype]
    return dict(campaign_id="validation", study="validation", implementation="cxx", benchmark=bench_name,
                benchmark_code=bench.code, dim=bench.dim, predictor_name=ptype, arm=ptype, predictor=pred,
                run_id=run_id, seed=seed, input_representation="raw", epsilon_fraction=cfg["epsilon_fraction"],
                xcsf=x, monitoring=dict(window=100, py_grid_checkpoints=2, py_unmatched="nearest"),
                grid_resolution=bench.grid_resolution, n_grid_points=len(bench.grid()))


def _run_cxx(binary: Path, workdir: Path, confsys_text: str):
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "confsys.xcsf").write_text(confsys_text)
    r = subprocess.run([str(binary), "-f", "xcsf"], cwd=workdir, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise ValidationError(f"xcsf-rf failed in {workdir}:\n{r.stderr[-2000:]}")


def test_cxx_benchmarks(binary: Path, log):
    from .parity import render_confsys
    for b in B.BENCHMARKS.values():
        spec = _make_spec(b.name)
        text = render_confsys(spec, "validation").replace("save problem execution trace = off",
                                                          "save problem execution trace = on")
        text = text.replace("evaluate solution = on", "evaluate solution = off").replace(
            "save action-value function = on", "save action-value function = off")
        with tempfile.TemporaryDirectory() as td:
            _run_cxx(binary, Path(td), text)
            with gzip.open(Path(td) / "trace.xcsf-0000.gz", "rt") as fh:
                lines = [l for l in fh.read().splitlines() if l.strip()]
        xs, ys = [], []
        for line in lines:
            if not (line.endswith("Learning") or line.endswith("Testing")):
                continue
            field = [f for f in line.split("\t") if ";" in f]
            _check(len(field) == 1, f"unexpected trace line '{line}'")
            vals = [float(v) for v in field[0].split(";")]
            xs.append(vals[:-1])
            ys.append(vals[-1])
        X, Y = np.array(xs), np.array(ys)
        _check(len(X) == 2 * spec["xcsf"]["n_learning_problems"], f"{b.name}: trace has {len(X)} problems")
        _check(np.all((X >= np.array(b.lower)) & (X <= np.array(b.upper))), f"{b.name}: xcslib inputs outside domain")
        # printed with 6 significant digits: a few values can print as integers by chance
        _check(np.mean(X != np.round(X)) > 0.9 and len(np.unique(X)) > 0.9 * X.size,
               f"{b.name}: xcslib inputs look discretised")
        ref = b(X)
        # trace values are printed with 6 significant digits
        tol = b.lipschitz * b.dim * 5e-6 * np.maximum(1, np.abs(X)).max(axis=1) + 5e-6 * np.maximum(1, np.abs(ref)) + 1e-9
        dev = np.abs(ref - Y)
        _check(np.all(dev <= tol), f"{b.name}: xcslib f(x) differs from the Python definition "
                                    f"(max dev {dev.max():.3g} at x={X[np.argmax(dev - tol)]})")
        log(f"  {b.name:20s} xcslib '{b.cxx_function}' matches Python on {len(X)} sampled points "
            f"(max dev {dev.max():.2e}, within print precision)")
    log("[ok] C++ benchmark functions agree with the Python definitions")


def test_cxx_rls_delta_run(binary: Path, log):
    """End-to-end: xcsf-rf selects rls_delta from confsys and reads x0/delta (short run, standard outputs)."""
    from .parity import render_confsys
    from .cxx import check_rls_delta_parsed, parse_population
    spec = _make_spec("sine_shifted_1d", "rls_delta", n=500)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "confsys.xcsf").write_text(render_confsys(spec, "validation"))
        r = subprocess.run([str(binary), "-f", "xcsf"], cwd=td, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise ValidationError(f"xcsf-rf with rls_delta failed:\n{r.stderr[-2000:]}")
        (td / "stderr.log").write_text(r.stderr)
        check_rls_delta_parsed(td, spec)
        pop = parse_population(td / "population.xcsf-0000.gz", 1, "rls_delta")
        _check(len(pop) > 0 and all(len(w) == 2 for w in pop["weights"]), "rls_delta population: unexpected weights")
    log("[ok] xcsf-rf runs with prediction function rls_delta (x0 and delta parsed, population readable)")


# ------------------------------------------------------------------------------------------- 5
def test_determinism(binary: Path | None, log):
    import sys
    from .config import PY_LIB_DIR
    sys.path.insert(0, str(PY_LIB_DIR / "src"))
    from . import python_runner
    spec = _make_spec("sine_shifted_1d", "nlms", n=1000)
    spec["implementation"] = "py"
    with tempfile.TemporaryDirectory() as td:
        a = python_runner.run(spec, Path(td) / "a")
        b = python_runner.run(spec, Path(td) / "b")
        for f in ("curve.csv", "grid_predictions.csv.gz", "population.csv.gz", "checkpoints.csv"):
            pa, pb = Path(td) / "a" / f, Path(td) / "b" / f
            ra = gzip.open(pa).read() if f.endswith(".gz") else pa.read_bytes()
            rb = gzip.open(pb).read() if f.endswith(".gz") else pb.read_bytes()
            _check(ra == rb, f"python run not deterministic: {f} differs")
        _check(a["training_inputs_sha256"] == b["training_inputs_sha256"], "python training stream not deterministic")
    log("[ok] python determinism (same seed -> identical raw outputs)")
    if binary is None:
        return
    from .parity import render_confsys
    spec["implementation"] = "cxx"
    text = render_confsys(spec, "validation")
    with tempfile.TemporaryDirectory() as td:
        for d in ("a", "b"):
            _run_cxx(binary, Path(td) / d, text)
        for f in ("statistics.xcsf-0000.gz", "avf.xcsf-0000.gz", "population.xcsf-0000.gz"):
            ra = gzip.open(Path(td) / "a" / f).read()
            rb = gzip.open(Path(td) / "b" / f).read()
            _check(ra == rb, f"xcslib run not deterministic: {f} differs")
    log("[ok] xcslib determinism (same seed -> identical standard result files)")


def run_all(out: Path, with_cxx=True, with_determinism=True):
    from .manifest import CXX_BIN, require_cxx_build, require_patch_applied, require_pf_driver
    out.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    test_benchmarks(log)
    plot_benchmarks(out, log)
    test_predictor_parity(log)
    binary = None
    if with_cxx:
        require_patch_applied()
        require_cxx_build()
        binary = CXX_BIN
        test_cxx_benchmarks(binary, log)
        test_cxx_predictors(require_pf_driver(), log)
        test_cxx_rls_delta_run(binary, log)
    else:
        log("[skip] C++ checks (--no-cxx)")
    if with_determinism:
        test_determinism(binary, log)
    (out / "validation_log.txt").write_text("\n".join(log_lines) + "\n")
    if with_cxx and with_determinism:
        # stamp checked by `plan`: the campaign can only be planned after a full validation
        # of exactly this build, these libraries and this execution code
        import json
        from . import manifest as M
        from .config import CXX_LIB_DIR, PY_LIB_DIR
        stamp = dict(validated_at=M.now(), binary_sha256=M.sha256_file(M.CXX_BIN),
                     xcslib=M.tree_hash(CXX_LIB_DIR)["sha256"], xcsf_python=M.tree_hash(PY_LIB_DIR)["sha256"],
                     execution_code=M.execution_code_hash()["sha256"],
                     reference_plots=sorted(p.name for p in out.glob("*.png")))
        (out / "validation_stamp.json").write_text(json.dumps(stamp, indent=2))
    else:
        stale = out / "validation_stamp.json"
        if stale.exists():
            stale.unlink()
        log("[warn] partial validation: no stamp written, planning stays blocked until a full validation passes")
    log(f"[ok] validation complete -> {out / 'validation_log.txt'}")
