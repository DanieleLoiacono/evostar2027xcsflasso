"""Semantic parameter parity between xcslib (C++) and xcsf_python.

This module is the only place where a resolved run spec is translated into
implementation-specific settings.  The translation is driven by the audit in
docs/PARAMETER_PARITY.md (source code was inspected, names were NOT assumed to
imply equal semantics).  Anything without a semantically identical counterpart
raises ``ParityError`` instead of being silently approximated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from . import benchmarks as B


class ParityError(ValueError):
    pass


# xcsf_python has ONE RLS ('rls', square-root information / QR form); both xcslib RLS variants are
# settings of it (docs/PARAMETER_PARITY.md, section 3):
#
# rls_xcslib: xcslib rls_pf never reads its static `delta` from the configuration (zero-initialised),
#   so V0 = 0, and it adds I to V after every update. Python: rls_delta = 0 (exact: the first
#   observation is ignored, as in xcslib where the gain is 0), process_noise = 1.
PY_RLS_XCSLIB_DELTA = 0.0
PY_RLS_XCSLIB_PROCESS_NOISE = 1.0
#
# rls_delta: the paper's RLS (Lanzi et al., IlliGAL 2005012, Sec. 7.3, Alg. 5: V0 = delta*I, no Q, no
#   forgetting).  xcslib: prediction function 'rls_delta' (patches/xcslib-rls-delta.patch), covariance
#   form.  Python: rls_delta = delta, process_noise = 0, forgetting_factor = 1, kalman_noise = False.
#   Same estimator, propagated as a triangular factor instead of a covariance (better rounding).
# Both settings: zero initial weights; offspring inherit the weights and restart from V0.

# xcslib parameters that are hard-coded or not settable through confsys
CXX_HARDCODED = {
    "delta": 0.1,            # xcsf_classifier_system::set_parameters: delta_del = 0.1
    "use_mam": True,         # "use MAM" is read but not in configuration_parameters -> cannot be set, default on
    "initial_set_size": 1.0, # python: covering always uses set_size=1.0 (not configurable)
}

_CXX_MUTATION = {"fixed": "fixed", "proportional": "proportional"}
_CXX_CROSSOVER = {"one_point": "one-point", "two_point": "two-point", "uniform": "uniform"}
_PY_SELECTION = {"roulette": "roulette"}
_CXX_SELECTION = {"roulette": "roulette-wheel"}


def _onoff(flag: bool) -> str:
    return "on" if flag else "off"


def _num(v) -> str:
    """Shortest round-trip representation, parsed identically by atof/strtod."""
    if isinstance(v, bool):
        raise TypeError("bool is not a number")
    if isinstance(v, int):
        return str(v)
    r = repr(float(v))
    return r[:-2] if r.endswith(".0") else r


def check_common(spec: Dict[str, Any], for_cxx: bool):
    x = spec["xcsf"]
    if x["condensation"]:
        raise ParityError("condensation is not part of this campaign (xcslib counts condensation problems, "
                          "xcsf_python counts condensation epochs: no 1:1 schedule)")
    if x["bounded"]:
        raise ParityError("bounded=True is not supported: xcsf_python derives the bounds from the first training "
                          "batch, xcslib from 'min/max input'")
    if x["selection"] not in _PY_SELECTION:
        raise ParityError(f"selection '{x['selection']}' not mapped")
    if x["initial_set_size"] != CXX_HARDCODED["initial_set_size"]:
        raise ParityError("xcsf_python always initialises the niche-size estimate of covered rules to 1.0")
    if for_cxx:
        for k in ("delta", "use_mam"):
            if x[k] != CXX_HARDCODED[k]:
                raise ParityError(f"xcslib hard-codes {k}={CXX_HARDCODED[k]!r}; got {x[k]!r}")
        if x["mutation"] not in _CXX_MUTATION:
            raise ParityError(f"xcslib mutation dispatcher does not support '{x['mutation']}'")
        if spec["input_representation"] != "raw":
            raise ParityError("xcslib runs must use raw inputs")


# ----------------------------------------------------------------------------------------- C++ ---

def cxx_sections(spec: Dict[str, Any]) -> List[Tuple[str, List[Tuple[str, str]], str]]:
    """Ordered (section, [(key, value)], comment) triples for confsys.<suffix>."""
    check_common(spec, for_cxx=True)
    bench = B.get(spec["benchmark"])
    x, p, mon = spec["xcsf"], spec["predictor"], spec["monitoring"]
    lo, hi = bench.lower[0], bench.upper[0]
    if any(l != lo for l in bench.lower) or any(u != hi for u in bench.upper):
        raise ParityError("xcslib supports a single [min input, max input] for all inputs")
    ptype = p["type"]
    pf = {"constant": "value", "nlms": "nlms", "rls_xcslib": "rls", "rls_delta": "rls_delta"}.get(ptype)
    if pf is None:
        raise ParityError(f"predictor type '{ptype}' has no xcslib counterpart")

    secs = []
    secs.append(("random", [("seed", _num(spec["seed"]))], "seed schedule: seed_base + run_id (shared with Python random_state)"))
    secs.append(("environment::real_functions", [
        ("function", bench.cxx_function),
        ("min input", _num(lo)),
        ("max input", _num(hi)),
        ("scale factor", _num(bench.cxx_scale_factor)),
        ("sampling resolution", _num(bench.grid_resolution)),
    ], f"benchmark {bench.name}; 'min/max input' require patches/xcslib-benchmark-functions.patch"))
    secs.append(("condition::real_interval", [
        ("input size", _num(bench.dim)),
        ("min input", _num(lo)),
        ("max input", _num(hi)),
        ("r0", _num(x["r0"])),
        ("m0", _num(x["m0"])),
        ("bounded", _onoff(x["bounded"])),
        ("mutation", _CXX_MUTATION[x["mutation"]]),
        ("crossover", _CXX_CROSSOVER[x["crossover"]]),
    ], "keys 'mutation'/'crossover' (NOT 'mutation method'/'crossover method', which xcslib silently ignores)"))
    secs.append(("classifier_system", [
        ("population size", _num(x["population_size"])),
        ("epsilon zero", _num(x["epsilon0"])),
        ("learning rate", _num(x["beta"])),
        ("alpha", _num(x["alpha"])),
        ("vi", _num(x["nu"])),
        ("theta GA", _num(x["theta_ga"])),
        ("crossover probability", _num(x["crossover_probability"])),
        ("mutation probability", _num(x["mutation_probability"])),
        ("theta delete", _num(x["theta_delete"])),
        ("theta GA sub", _num(x["theta_subsume"])),
        ("GA subsumption", _onoff(x["ga_subsumption"])),
        ("GA subsumption on [A]", _onoff(x["ga_subsumption"])),
        ("AS subsumption", _onoff(x["as_subsumption"])),
        ("theta AS sub", "100"),
        ("offspring selection for GA", _CXX_SELECTION[x["selection"]]),
        ("error init", _num(x["initial_error"])),
        ("fitness init", _num(x["initial_fitness"])),
        ("set size init", _num(x["initial_set_size"])),
        ("prediction init", "0"),
        ("update error first", _onoff(x["error_before_prediction"])),
        ("update during test", "off"),
        ("discovery component", "on"),
        ("initial population", "empty"),
        ("exploration strategy", "random"),
    ], "'update during test' MUST be off (xcslib default is on; Python never learns at prediction time)"))
    secs.append(("experiments", [
        ("first experiment", _num(spec["run_id"])),
        ("number of experiments", "1"),
        ("first problem", "0"),
        ("number of learning problems", _num(x["n_learning_problems"])),
        ("number of condensation problems", "0"),
        ("statistics rolling window size", _num(mon["window"])),
        ("evaluate solution", "on"),
        ("save action-value function", "on"),
        ("save final population", "on"),
        ("save population every", "0"),
        ("save execution time report", "on"),
        ("save problem execution trace", "off"),
    ], "one experiment per process: run_id == 'first experiment', fresh RNG seeded with the run seed"))
    secs.append(("prediction::base", [
        ("input size", _num(bench.dim)),
        ("prediction function", pf),
        ("degree", "1"),
    ], ""))
    nlms_eta = p["eta"] if ptype == "nlms" else 0.2
    nlms_x0 = p["x0"] if ptype == "nlms" else 1.0
    secs.append(("prediction::nlms", [("learning rate", _num(nlms_eta)), ("x0", _num(nlms_x0))],
                 "" if ptype == "nlms" else "inert: xcslib refuses to start without this section"))
    if ptype == "constant":
        secs.append(("prediction::value", [("learning rate", _num(p["eta"]))],
                     "value_pf: w <- w + eta (y - w); no MAM on the predictor"))
    if ptype == "rls_xcslib":
        secs.append(("prediction::rls", [("x0", _num(p["x0"]))],
                     "rls_pf: V0 = delta*I with delta never read (=0); V <- V - K phi^T V + I"))
    if ptype == "rls_delta":
        if not (float(p["delta"]) > 0):
            raise ParityError("rls_delta requires delta > 0")
        secs.append(("prediction::rls_delta", [("x0", _num(p["x0"])), ("delta", _num(p["delta"]))],
                     "rls_delta_pf (patches/xcslib-rls-delta.patch): V0 = delta*I; V <- V - (V phi)(V phi)^T/(1+phi^T V phi)"))
    return secs


def render_confsys(spec: Dict[str, Any], spec_hash: str) -> str:
    lines = [f"// generated by campaign/xcsfcamp ({spec['campaign_id']}) - do not edit",
             f"// run {spec['study']}/{spec['implementation']}/{spec['benchmark']}/{spec['arm']}/run_{spec['run_id']:04d}",
             f"// spec_hash {spec_hash}"]
    for sec, kv, comment in cxx_sections(spec):
        if comment:
            lines.append(f"// {comment}")
        lines.append(f"<{sec}>")
        for k, v in kv:
            lines.append(f"\t{k} = {v}")
        lines.append(f"</{sec}>")
        lines.append("")
    return "\n".join(lines)


# Values expected in the output of `xcsf-rf -f <suffix> -p` (what the binary actually parsed).
def cxx_expected_print(spec: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
    secs = {s: dict(kv) for s, kv, _ in cxx_sections(spec)}
    cs, cond, env, ex = secs["classifier_system"], secs["condition::real_interval"], secs["environment::real_functions"], secs["experiments"]
    exp = {
        ("random", "seed"): int(secs["random"]["seed"]),
        ("environment::real_functions", "min value"): float(env["min input"]),
        ("environment::real_functions", "max value"): float(env["max input"]),
        ("environment::real_functions", "scale factor"): float(env["scale factor"]),
        ("environment::real_functions", "sampling resolution"): float(env["sampling resolution"]),
        ("condition::real_interval", "input size"): int(cond["input size"]),
        ("condition::real_interval", "min input"): float(cond["min input"]),
        ("condition::real_interval", "max input"): float(cond["max input"]),
        ("condition::real_interval", "r0"): float(cond["r0"]),
        ("condition::real_interval", "m0"): float(cond["m0"]),
        ("condition::real_interval", "bounded"): cond["bounded"],
        ("condition::real_interval", "mutation"): cond["mutation"],
        ("condition::real_interval", "crossover"): cond["crossover"],
        ("classifier_system", "population size"): int(cs["population size"]),
        ("classifier_system", "epsilon zero"): float(cs["epsilon zero"]),
        ("classifier_system", "theta GA"): float(cs["theta GA"]),
        ("classifier_system", "crossover probability"): float(cs["crossover probability"]),
        ("classifier_system", "mutation probability"): float(cs["mutation probability"]),
        ("classifier_system", "learning rate"): float(cs["learning rate"]),
        ("classifier_system", "discovery component"): "on",
        ("classifier_system", "vi"): float(cs["vi"]),
        ("classifier_system", "alpha"): float(cs["alpha"]),
        ("classifier_system", "error init"): float(cs["error init"]),
        ("classifier_system", "fitness init"): float(cs["fitness init"]),
        ("classifier_system", "set size init"): float(cs["set size init"]),
        ("classifier_system", "theta delete"): float(cs["theta delete"]),
        ("classifier_system", "theta GA sub"): float(cs["theta GA sub"]),
        ("classifier_system", "GA subsumption"): cs["GA subsumption"],
        ("classifier_system", "GAA subsumption"): cs["GA subsumption on [A]"],
        ("classifier_system", "AS subsumption"): cs["AS subsumption"],
        ("classifier_system", "update during test"): "off",
        ("classifier_system", "update error first"): cs["update error first"],
        ("classifier_system", "use MAM"): "on",
        ("experiments", "first experiment"): int(ex["first experiment"]),
        ("experiments", "number of experiments"): 1,
        ("experiments", "number of learning problems"): int(ex["number of learning problems"]),
        ("experiments", "number of condensation problems"): 0,
        ("experiments", "evaluate solution"): "on",
        ("experiments", "save action-value function"): "on",
        ("experiments", "save final population"): "on",
    }
    return exp


def parse_cxx_print(text: str) -> Dict[Tuple[str, str], str]:
    """Parse `xcsf-rf -p` output (<section> ... key = value ...)."""
    out, section = {}, None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("</"):
            section = None
            continue
        if s.startswith("<") and s.endswith(">") and not s.startswith("<<"):
            section = s[1:-1]
            continue
        if section and "=" in s:
            k, v = s.split("=", 1)
            out.setdefault((section, k.strip()), v.strip())
    return out


def verify_cxx_print(spec: Dict[str, Any], printed_text: str) -> List[str]:
    """Return a list of mismatches between intended and effectively parsed C++ parameters."""
    got = parse_cxx_print(printed_text)
    problems = []
    for key, want in cxx_expected_print(spec).items():
        if key not in got:
            problems.append(f"{key}: not printed by xcslib")
            continue
        g = got[key]
        if isinstance(want, str):
            if g != want:
                problems.append(f"{key}: expected '{want}', xcslib parsed '{g}'")
        else:
            try:
                gv = float(g)
            except ValueError:
                problems.append(f"{key}: non-numeric '{g}'")
                continue
            # xcslib prints most values with 2 (sometimes 5) decimals
            tol = 0.0051 + 1e-9 * abs(want)
            if abs(gv - float(want)) > tol:
                problems.append(f"{key}: expected {want}, xcslib parsed {g}")
    return problems


# -------------------------------------------------------------------------------------- Python ---

def py_params(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword arguments for xcsf.XCSFRegressor (explicit, no implicit defaults)."""
    check_common(spec, for_cxx=False)
    x, p, mon = spec["xcsf"], spec["predictor"], spec["monitoring"]
    t = p["type"]
    kw: Dict[str, Any] = dict(
        population_size=int(x["population_size"]), n_epochs=1, degree=1,
        learning_rate=float(x["beta"]), epsilon_0=float(x["epsilon0"]), alpha=float(x["alpha"]), nu=float(x["nu"]),
        theta_ga=x["theta_ga"], crossover_probability=float(x["crossover_probability"]),
        mutation_probability=float(x["mutation_probability"]), cover_radius=float(x["r0"]),
        mutation_scale=float(x["m0"]), mutation=x["mutation"], crossover=x["crossover"],
        theta_delete=x["theta_delete"], delta=float(x["delta"]), theta_subsume=x["theta_subsume"],
        ga_subsumption=bool(x["ga_subsumption"]), match_subsumption=bool(x["as_subsumption"]),
        theta_match_subsume=100, selection=_PY_SELECTION[x["selection"]], tournament_fraction=0.4,
        initial_fitness=float(x["initial_fitness"]), initial_error=float(x["initial_error"]),
        initial_prediction=0.0, use_mam=bool(x["use_mam"]),
        error_before_prediction=bool(x["error_before_prediction"]),
        normalize=False, bounded=False, discovery=True, condensation_epochs=0, niche_history=0,
        history_interval=int(mon["window"]), unmatched=mon["py_unmatched"], shuffle=False,
        random_state=int(spec["seed"]),
        # predictor defaults (overwritten below for the active predictor; inert otherwise)
        prediction_learning_rate=0.2, x0=1.0, rls_delta=1000.0, forgetting_factor=1.0, process_noise=0.0,
        kalman_noise=False, lasso_alpha=0.001, lasso_window=256, lasso_max_iter=1000, lasso_tol=1e-6,
        lasso_learning_rate_decay=0.0,
    )
    if t == "constant":
        kw.update(prediction="constant", prediction_learning_rate=float(p["eta"]))
    elif t == "nlms":
        kw.update(prediction="nlms", prediction_learning_rate=float(p["eta"]), x0=float(p["x0"]))
    elif t == "rls_xcslib":
        kw.update(prediction="rls", x0=float(p["x0"]), rls_delta=PY_RLS_XCSLIB_DELTA, forgetting_factor=1.0,
                  process_noise=PY_RLS_XCSLIB_PROCESS_NOISE, kalman_noise=False)
    elif t == "rls_delta":
        if not (float(p["delta"]) > 0):
            raise ParityError("rls_delta requires delta > 0")
        kw.update(prediction="rls", x0=float(p["x0"]), rls_delta=float(p["delta"]), forgetting_factor=1.0,
                  process_noise=0.0, kalman_noise=False)
    elif t == "lasso_online":
        # recursive Lasso on RLS statistics: no learning rate; delta/forgetting as for rls_delta
        kw.update(prediction="lasso_online", x0=float(p["x0"]), lasso_alpha=float(p["lasso_alpha"]),
                  rls_delta=float(p["delta"]), forgetting_factor=float(p["forgetting_factor"]),
                  lasso_max_iter=int(p["max_iter"]), lasso_tol=float(p["tol"]))
    elif t == "lasso_sgd":
        kw.update(prediction="lasso_sgd", prediction_learning_rate=float(p["eta"]), x0=float(p["x0"]),
                  lasso_alpha=float(p["lasso_alpha"]), lasso_learning_rate_decay=float(p["learning_rate_decay"]))
    elif t == "lasso_batch":
        kw.update(prediction="lasso_batch", x0=float(p["x0"]), lasso_alpha=float(p["lasso_alpha"]),
                  lasso_window=None if p["window"] is None else int(p["window"]),
                  lasso_max_iter=int(p["max_iter"]), lasso_tol=float(p["tol"]))
    else:
        raise ParityError(f"unknown predictor type {t}")
    return kw


# ---------------------------------------------------------------------------------- audit table ---

def audit_rows(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Side-by-side table of every semantic parameter for one (benchmark, predictor) parity cell."""
    secs = {s: dict(kv) for s, kv, _ in cxx_sections(spec)}
    kw = py_params(spec)
    cs, cond = secs["classifier_system"], secs["condition::real_interval"]
    rows = []

    def row(name, cxx_loc, cxx_val, py_name, py_val, equal=None, note=""):
        if equal is None:
            try:
                equal = abs(float(cxx_val) - float(py_val)) <= 1e-12 * max(1.0, abs(float(py_val)))
            except (TypeError, ValueError):
                equal = str(cxx_val) == str(py_val)
        rows.append(dict(parameter=name, cxx=f"{cxx_loc} = {cxx_val}", python=f"{py_name} = {py_val}",
                         equal=bool(equal), note=note))

    row("population size N (micro)", "population size", cs["population size"], "population_size", kw["population_size"])
    row("epsilon0", "epsilon zero", cs["epsilon zero"], "epsilon_0", kw["epsilon_0"])
    row("beta (error, fitness, niche size)", "learning rate", cs["learning rate"], "learning_rate", kw["learning_rate"])
    row("alpha (accuracy)", "alpha", cs["alpha"], "alpha", kw["alpha"])
    row("nu (accuracy exponent)", "vi", cs["vi"], "nu", kw["nu"])
    row("theta_GA", "theta GA", cs["theta GA"], "theta_ga", kw["theta_ga"],
        note="both compare time minus numerosity-weighted mean timestamp; time = learning steps only")
    row("chi (crossover prob.)", "crossover probability", cs["crossover probability"], "crossover_probability", kw["crossover_probability"])
    row("mu (mutation prob. per endpoint)", "mutation probability", cs["mutation probability"], "mutation_probability", kw["mutation_probability"])
    row("r0 (cover radius)", "r0", cond["r0"], "cover_radius", kw["cover_radius"], note="raw input units, normalize=False")
    row("m0 (mutation scale)", "m0", cond["m0"], "mutation_scale", kw["mutation_scale"],
        note="fixed: U(-m0,m0) in both; xcslib bug: upper endpoint set to the new LOWER (see PARAMETER_PARITY.md)")
    row("mutation operator", "mutation", cond["mutation"], "mutation", kw["mutation"], equal=True)
    row("crossover operator", "crossover", cond["crossover"], "crossover", kw["crossover"], equal=True)
    row("bounded conditions", "bounded", cond["bounded"], "bounded", kw["bounded"], equal=(cond["bounded"] == "off") == (not kw["bounded"]),
        note="xcslib check() still clips to [min,max] after an upper-endpoint crossover")
    row("theta_del", "theta delete", cs["theta delete"], "theta_delete", kw["theta_delete"])
    row("delta (deletion)", "(hard-coded)", CXX_HARDCODED["delta"], "delta", kw["delta"])
    row("theta_sub", "theta GA sub", cs["theta GA sub"], "theta_subsume", kw["theta_subsume"])
    row("GA subsumption (parents, then [A])", "GA subsumption / on [A]", cs["GA subsumption"] + "/" + cs["GA subsumption on [A]"],
        "ga_subsumption", kw["ga_subsumption"], equal=(cs["GA subsumption"] == "on") == kw["ga_subsumption"] and cs["GA subsumption"] == cs["GA subsumption on [A]"])
    row("action/match-set subsumption", "AS subsumption", cs["AS subsumption"], "match_subsumption", kw["match_subsumption"],
        equal=(cs["AS subsumption"] == "on") == kw["match_subsumption"], note="xcslib never calls it from step(); keep off")
    row("selection", "offspring selection for GA", cs["offspring selection for GA"], "selection", kw["selection"], equal=True)
    row("initial fitness", "fitness init", cs["fitness init"], "initial_fitness", kw["initial_fitness"])
    row("initial error", "error init", cs["error init"], "initial_error", kw["initial_error"])
    row("initial niche size", "set size init", cs["set size init"], "(hard-coded)", 1.0)
    row("MAM", "(not settable, on)", "on", "use_mam", kw["use_mam"], equal=kw["use_mam"] is True)
    row("error updated before prediction", "update error first", cs["update error first"], "error_before_prediction",
        kw["error_before_prediction"], equal=(cs["update error first"] == "on") == kw["error_before_prediction"])
    row("learning during evaluation", "update during test", cs["update during test"], "(predict never learns)", "off", equal=cs["update during test"] == "off")
    row("training budget", "number of learning problems", secs["experiments"]["number of learning problems"],
        "len(X_train), n_epochs=1, shuffle=False", spec["xcsf"]["n_learning_problems"])
    row("seed", "<random> seed", secs["random"]["seed"], "random_state", kw["random_state"],
        note="same integer; RNG engines differ (mt19937_64 vs numpy MT19937): streams are NOT shared")
    pt = spec["predictor"]["type"]
    pb = secs["prediction::base"]
    if pt == "constant":
        row("predictor", "prediction function", pb["prediction function"], "prediction", kw["prediction"], equal=True)
        row("eta (constant)", "prediction::value learning rate", secs["prediction::value"]["learning rate"],
            "prediction_learning_rate", kw["prediction_learning_rate"])
        row("initial predictor value", "(uninitialised member in value_pf)", "~0", "initial_prediction", 0.0, equal=True,
            note="xcslib leaves value_pf::prediction uninitialised (UB); populations are checked for absurd values")
    elif pt == "nlms":
        row("predictor", "prediction function", pb["prediction function"], "prediction", kw["prediction"], equal=True)
        row("eta (NLMS)", "prediction::nlms learning rate", secs["prediction::nlms"]["learning rate"],
            "prediction_learning_rate", kw["prediction_learning_rate"])
        row("x0", "prediction::nlms x0", secs["prediction::nlms"]["x0"], "x0", kw["x0"])
    elif pt == "rls_xcslib":
        row("predictor", "prediction function", pb["prediction function"], "prediction", kw["prediction"], equal=True,
            note="xcslib 'rls' == Kalman filter with V0=0, lambda=1, R=1, Q=I; Python 'rls' in the same setting")
        row("x0", "prediction::rls x0", secs["prediction::rls"]["x0"], "x0", kw["x0"])
        row("V0 scale", "(delta never read -> 0)", 0.0, "rls_delta", kw["rls_delta"],
            note="exact: with zero covariance the first observation is ignored in both")
        row("process noise Q", "(V += I every update)", 1.0, "process_noise", kw["process_noise"])
        row("forgetting factor", "(none)", 1.0, "forgetting_factor", kw["forgetting_factor"])
        row("measurement noise R", "(1 in 1 + phi^T V phi)", "off", "kalman_noise", "off" if not kw["kalman_noise"] else "on")
    elif pt == "rls_delta":
        row("predictor", "prediction function", pb["prediction function"], "prediction", kw["prediction"], equal=True,
            note="paper RLS (Alg. 5): xcslib rls_delta_pf (covariance form) vs Python rls with Q=0, lambda=1, R=1 "
                 "(square-root information form of the same recursion)")
        row("x0", "prediction::rls_delta x0", secs["prediction::rls_delta"]["x0"], "x0", kw["x0"])
        row("delta (V0 = delta*I)", "prediction::rls_delta delta", secs["prediction::rls_delta"]["delta"],
            "rls_delta", kw["rls_delta"])
        row("process noise Q", "(none)", 0.0, "process_noise", kw["process_noise"])
        row("forgetting factor", "(none)", 1.0, "forgetting_factor", kw["forgetting_factor"])
        row("measurement noise R", "(1 in beta = 1 + phi^T V phi)", "off", "kalman_noise", "off" if not kw["kalman_noise"] else "on")
    return rows
