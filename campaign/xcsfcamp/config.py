"""Campaign configuration: loading, strict validation and expansion into run specs.

Every run is described by a fully *resolved* spec (plain JSON, no defaults left
implicit).  Its SHA-256 (``spec_hash``) is stored next to the raw results and is
what makes resume safe: a completed run is skipped only when its stored hash is
identical to the planned one, otherwise the tools stop with an error.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from . import benchmarks as B

CAMPAIGN_DIR = Path(__file__).resolve().parents[1]          # .../campaign
PROJECT_DIR = CAMPAIGN_DIR.parent                            # repository root
CXX_LIB_DIR = PROJECT_DIR / "xcslib-1.5-rc1-niches"
PY_LIB_DIR = PROJECT_DIR / "xcsf_python-2.0.0"

ALLOWED_TOP = {"campaign_id", "description", "n_runs", "seed_base", "epsilon_fraction", "benchmarks", "xcsf",
               "benchmark_overrides", "predictors", "studies", "monitoring", "analysis", "profiles"}
ALLOWED_XCSF = {"population_size", "n_learning_problems", "beta", "alpha", "nu", "theta_ga",
                "crossover_probability", "mutation_probability", "r0_fraction", "m0_fraction", "mutation",
                "crossover", "selection", "theta_delete", "delta", "theta_subsume", "ga_subsumption",
                "as_subsumption", "initial_fitness", "initial_error", "initial_set_size", "use_mam",
                "error_before_prediction", "bounded", "condensation"}
PREDICTOR_KEYS = {
    "constant": {"type", "eta"},
    "nlms": {"type", "eta", "x0"},
    "rls_xcslib": {"type", "x0"},
    "rls_standard": {"type", "x0", "delta"},
    "lasso_online": {"type", "eta", "x0", "lasso_alpha", "learning_rate_decay"},
    "lasso_batch": {"type", "x0", "lasso_alpha", "window", "max_iter", "tol"},
}
CXX_SUPPORTED_TYPES = {"constant", "nlms", "rls_xcslib"}
ALLOWED_STUDY = {"description", "implementations", "predictors", "input_representation", "benchmarks"}
ALLOWED_MONITORING = {"window", "py_grid_checkpoints", "py_unmatched"}
ALLOWED_ANALYSIS = {"alpha", "n_bootstrap", "last_fraction", "convergence_consecutive_windows",
                    "equivalence_margin_error_eps_fraction", "equivalence_margin_size_relative",
                    "equivalence_margin_fraction_abs", "bootstrap_seed"}


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _check_keys(where: str, d: dict, allowed: set):
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"{where}: unsupported key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def load_config(path: Path, profile: str | None = None) -> Dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    _check_keys("config", raw, ALLOWED_TOP)
    cfg = {k: v for k, v in raw.items() if k != "profiles"}
    if profile:
        profiles = raw.get("profiles", {})
        if profile not in profiles:
            raise ConfigError(f"unknown profile '{profile}'; available: {sorted(profiles)}")
        prof = dict(profiles[profile])
        suffix = prof.pop("campaign_id_suffix", f"-{profile}")
        _check_keys(f"profiles.{profile}", prof, ALLOWED_TOP - {"profiles", "campaign_id"})
        cfg = _deep_merge(cfg, prof)
        cfg["campaign_id"] = raw["campaign_id"] + suffix
    cfg["profile"] = profile or "full"
    validate(cfg)
    return cfg


def validate(cfg: dict):
    for k in ("campaign_id", "n_runs", "seed_base", "epsilon_fraction", "benchmarks", "xcsf", "predictors", "studies",
              "monitoring", "analysis"):
        if k not in cfg:
            raise ConfigError(f"missing required key '{k}'")
    if not isinstance(cfg["n_runs"], int) or cfg["n_runs"] < 1:
        raise ConfigError("n_runs must be a positive integer")
    if not isinstance(cfg["seed_base"], int) or cfg["seed_base"] < 1:
        raise ConfigError("seed_base must be a positive integer (xcslib treats seed 0 as 'seed from clock')")
    if not (0 < cfg["epsilon_fraction"] < 1):
        raise ConfigError("epsilon_fraction must be in (0,1)")
    for b in cfg["benchmarks"]:
        B.get(b)
    _check_keys("xcsf", cfg["xcsf"], ALLOWED_XCSF)
    missing = ALLOWED_XCSF - set(cfg["xcsf"])
    if missing:
        raise ConfigError(f"xcsf: every parameter must be explicit; missing {sorted(missing)}")
    for b, ov in cfg.get("benchmark_overrides", {}).items():
        B.get(b)
        _check_keys(f"benchmark_overrides.{b}", ov, ALLOWED_XCSF)
    for name, p in cfg["predictors"].items():
        t = p.get("type")
        if t not in PREDICTOR_KEYS:
            raise ConfigError(f"predictor '{name}': unknown type '{t}'; known {sorted(PREDICTOR_KEYS)}")
        _check_keys(f"predictors.{name}", p, PREDICTOR_KEYS[t])
        missing = PREDICTOR_KEYS[t] - set(p)
        if missing:
            raise ConfigError(f"predictor '{name}': missing {sorted(missing)}")
    for sname, s in cfg["studies"].items():
        _check_keys(f"studies.{sname}", s, ALLOWED_STUDY)
        if s["input_representation"] not in ("raw", "unit"):
            raise ConfigError(f"studies.{sname}.input_representation must be 'raw' or 'unit'")
        for impl in s["implementations"]:
            if impl not in ("cxx", "py"):
                raise ConfigError(f"studies.{sname}: unknown implementation '{impl}'")
        if "cxx" in s["implementations"] and s["input_representation"] != "raw":
            raise ConfigError(f"studies.{sname}: xcslib has no input scaling, a study including 'cxx' must use 'raw'")
        for b in s.get("benchmarks", []):   # optional per-study filter of the campaign benchmark list
            B.get(b)
        for p in s["predictors"]:
            if p not in cfg["predictors"]:
                raise ConfigError(f"studies.{sname}: unknown predictor '{p}'")
            if "cxx" in s["implementations"] and cfg["predictors"][p]["type"] not in CXX_SUPPORTED_TYPES:
                raise ConfigError(f"studies.{sname}: predictor '{p}' ({cfg['predictors'][p]['type']}) has no "
                                  f"semantically identical xcslib counterpart; keep it in a Python-only study")
    _check_keys("monitoring", cfg["monitoring"], ALLOWED_MONITORING)
    if cfg["monitoring"]["window"] % 100 != 0:
        raise ConfigError("monitoring.window must be a multiple of 100 (xcslib 'statistics rolling window size')")
    if cfg["monitoring"]["py_unmatched"] not in ("nearest", "mean", "raise"):
        raise ConfigError("monitoring.py_unmatched must be nearest|mean|raise")
    _check_keys("analysis", cfg["analysis"], ALLOWED_ANALYSIS)


def _expand_variants(name: str, p: dict) -> List[tuple]:
    """List-valued predictor parameters (e.g. lasso_alpha) expand into variants."""
    list_keys = sorted(k for k, v in p.items() if isinstance(v, list))
    if not list_keys:
        return [(name, dict(p))]
    out = []
    for combo in itertools.product(*[p[k] for k in list_keys]):
        q = dict(p)
        q.update(dict(zip(list_keys, combo)))
        label = name + "".join(f"_{_short(k)}{_fmt(v)}" for k, v in zip(list_keys, combo))
        out.append((label, q))
    return out


def _short(k):
    return {"lasso_alpha": "a", "window": "w", "eta": "eta"}.get(k, k)


def _fmt(v):
    return repr(v).replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class RunSpec:
    data: Dict[str, Any]

    @property
    def key(self) -> str:
        d = self.data
        return f"{d['study']}/{d['implementation']}/{d['benchmark']}/{d['arm']}/run_{d['run_id']:04d}"

    @property
    def spec_hash(self) -> str:
        return sha256_json(self.data)

    def __getitem__(self, k):
        return self.data[k]


def resolve_xcsf(cfg: dict, bench: B.Benchmark, representation: str) -> Dict[str, Any]:
    x = dict(cfg["xcsf"])
    x.update(cfg.get("benchmark_overrides", {}).get(bench.name, {}))
    width = bench.width
    if representation == "raw":
        x["r0"] = x["r0_fraction"] * width
        x["m0"] = x["m0_fraction"] * width
        x["domain_lower"], x["domain_upper"] = list(bench.lower), list(bench.upper)
    else:  # unit: z = (x - lower) / width in [0, 1]
        x["r0"] = x["r0_fraction"]
        x["m0"] = x["m0_fraction"]
        x["domain_lower"], x["domain_upper"] = [0.0] * bench.dim, [1.0] * bench.dim
    x["epsilon0"] = bench.epsilon0(cfg["epsilon_fraction"])
    return x


def expand_runs(cfg: dict) -> List[RunSpec]:
    runs = []
    for sname, study in cfg["studies"].items():
        for bname in [b for b in cfg["benchmarks"] if b in study.get("benchmarks", cfg["benchmarks"])]:
            bench = B.get(bname)
            xcsf = resolve_xcsf(cfg, bench, study["input_representation"])
            for pname in study["predictors"]:
                for arm, pred in _expand_variants(pname, cfg["predictors"][pname]):
                    for impl in study["implementations"]:
                        for r in range(cfg["n_runs"]):
                            runs.append(RunSpec({
                                "campaign_id": cfg["campaign_id"],
                                "study": sname,
                                "implementation": impl,
                                "benchmark": bname,
                                "benchmark_code": bench.code,
                                "dim": bench.dim,
                                "predictor_name": pname,
                                "arm": arm,
                                "predictor": pred,
                                "run_id": r,
                                "seed": cfg["seed_base"] + r,
                                "input_representation": study["input_representation"],
                                "epsilon_fraction": cfg["epsilon_fraction"],
                                "xcsf": xcsf,
                                "monitoring": cfg["monitoring"],
                                "grid_resolution": bench.grid_resolution,
                                "n_grid_points": int(len(bench.grid())),
                            }))
    keys = [r.key for r in runs]
    if len(keys) != len(set(keys)):
        raise ConfigError("duplicate run keys after expansion")
    return runs


def results_dir(cfg: dict, root: Path | None = None) -> Path:
    return (root or CAMPAIGN_DIR / "results") / cfg["campaign_id"]
