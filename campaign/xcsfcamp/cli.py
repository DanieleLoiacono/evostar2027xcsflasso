"""Command line entry point:  python -m xcsfcamp <command> ...

Commands are deliberately small and composable; the numbered shell scripts in
campaign/scripts/ chain them in the intended order.
"""

from __future__ import annotations

import os

# one BLAS thread per process: parallelism comes from running several runs at once
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import CAMPAIGN_DIR, PY_LIB_DIR, PY_LIB_VERSION, load_config, results_dir

# Use the upstream xcsf_python sources in this repository (not some other installed copy).
sys.path.insert(0, str(PY_LIB_DIR / "src"))

DEFAULT_CONFIG = CAMPAIGN_DIR / "config" / "campaign.json"


def _cdir(args) -> Path:
    if getattr(args, "campaign_dir", None):
        return Path(args.campaign_dir).resolve()
    cfg = load_config(Path(args.config), args.profile)
    return results_dir(cfg, Path(args.root).resolve() if args.root else None)


def _check_xcsf_import():
    import xcsf
    path = Path(xcsf.__file__).resolve()
    if PY_LIB_DIR.resolve() not in path.parents:
        raise SystemExit(f"[fail] imported xcsf from {path}, expected the copy in {PY_LIB_DIR}")
    if getattr(xcsf, "__version__", None) != PY_LIB_VERSION:
        raise SystemExit(f"[fail] xcsf version {getattr(xcsf, '__version__', None)} != {PY_LIB_VERSION}")
    return xcsf


# ------------------------------------------------------------------------------------------------ plan

def cmd_plan(args):
    from .manifest import VALIDATION_DIR, require_patch_applied, require_validation
    from .runs import campaign_paths, write_plan
    require_patch_applied()
    _check_xcsf_import()
    stamp = require_validation()
    cfg = load_config(Path(args.config), args.profile)
    cdir = results_dir(cfg, Path(args.root).resolve() if args.root else None)
    info = write_plan(cfg, cdir)
    vdir = campaign_paths(cdir)["manifest"] / "validation"
    if not vdir.exists():   # keep the validation evidence (log, stamp, reference plots) with the campaign
        shutil.copytree(VALIDATION_DIR, vdir)
    print(f"[ok] validation of {stamp['validated_at']} recorded in {vdir}")
    print(f"[ok] campaign {cfg['campaign_id']} -> {cdir}")
    print(f"     planned runs: {info['total']} (new {info['added']}, already planned {info['existing']})")
    print(cdir)


def cmd_where(args):
    print(_cdir(args))


def cmd_audit(args):
    """Parameter parity table for every parity cell (benchmark x predictor)."""
    import pandas as pd
    from .parity import audit_rows
    from .runs import campaign_paths, load_plan
    cdir = _cdir(args)
    plan = load_plan(cdir)
    rows, seen = [], set()
    for rec in plan.values():
        s = rec["spec"]
        if s["implementation"] != "cxx" or s["run_id"] != 0:
            continue
        cell = (s["study"], s["benchmark"], s["arm"])
        if cell in seen:
            continue
        seen.add(cell)
        for r in audit_rows(s):
            rows.append(dict(study=s["study"], benchmark=s["benchmark"], predictor=s["arm"], **r))
    if not rows:
        print("[info] no C++ runs planned; nothing to audit")
        return
    df = pd.DataFrame(rows)
    out = campaign_paths(cdir)["manifest"]
    df.to_csv(out / "parity_audit.csv", index=False)
    bad = df[~df["equal"]]
    with open(out / "parity_audit.md", "w") as fh:
        first = df[(df.benchmark == df.benchmark.iloc[0])]
        for pred, g in first.groupby("predictor", sort=False):
            fh.write(f"\n### {g.benchmark.iloc[0]} / {pred}\n\n| parameter | xcslib | xcsf_python | equal | note |\n|---|---|---|---|---|\n")
            for _, r in g.iterrows():
                fh.write(f"| {r.parameter} | `{r.cxx}` | `{r.python}` | {'yes' if r.equal else '**NO**'} | {r.note} |\n")
    if len(bad):
        print(bad.to_string())
        raise SystemExit(f"[fail] {len(bad)} parameter mismatch(es) between implementations (see {out/'parity_audit.csv'})")
    print(f"[ok] parity audit: {len(df)} checks, all equal -> {out/'parity_audit.md'}")


# -------------------------------------------------------------------------------------------- python

_LOCK = None


def _acquire_lock(cdir: Path, label: str, force: bool):
    global _LOCK
    lock = cdir / "raw" / f".lock-{label}"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if force and lock.exists():
        lock.unlink()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"[fail] {lock} exists: another {label} runner is active (or crashed). "
                         "If no runner is active, rerun with --break-lock.")
    os.write(fd, f"pid={os.getpid()} started={time.ctime()}\n".encode())
    os.close(fd)
    _LOCK = lock


def _release_lock():
    if _LOCK and _LOCK.exists():
        _LOCK.unlink()


def _py_worker(cdir_str: str, rec: dict):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(PY_LIB_DIR / "src"))
    from . import python_runner
    from .runs import publish, tmp_dir
    cdir = Path(cdir_str)
    t = tmp_dir(cdir, rec["key"])
    t0 = time.time()
    res = python_runner.run(rec["spec"], t)
    publish(cdir, rec, dict(implementation="py", wall_time_s=time.time() - t0,
                            train_time_s=res["train_time_s"], final_macro=res["final_macro"]))
    return rec["key"], time.time() - t0


def cmd_run_python(args):
    from .runs import load_plan, quarantine, select, session_snapshot, state
    _check_xcsf_import()
    cdir = _cdir(args)
    plan = load_plan(cdir)
    recs = select(plan, "py", args.study, args.benchmark, args.arm)
    _acquire_lock(cdir, "py", args.break_lock)
    try:
        session_snapshot(cdir, "py")
        pending = []
        for rec in sorted(recs, key=lambda r: (r["spec"]["run_id"], r["key"])):
            st = state(cdir, rec)
            if st == "incomplete":
                quarantine(cdir, rec["key"])
                st = "pending"
            if st == "pending":
                pending.append(rec)
        if args.limit:
            pending = pending[: args.limit]
        print(f"[info] python runs: {len(recs)} selected, {len(pending)} to run, jobs={args.jobs}", flush=True)
        failures = 0
        t_start = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_py_worker, str(cdir), rec): rec["key"] for rec in pending}
            for i, f in enumerate(as_completed(futs), 1):
                key = futs[f]
                try:
                    _, dt = f.result()
                    eta = (time.time() - t_start) / i * (len(pending) - i)
                    print(f"[done {i}/{len(pending)}] {key} ({dt:.1f}s, eta {eta/60:.1f} min)", flush=True)
                except Exception:
                    failures += 1
                    print(f"[FAIL {i}/{len(pending)}] {key}\n{traceback.format_exc()}", flush=True)
        if failures:
            raise SystemExit(f"[fail] {failures} python run(s) failed; their partial outputs are kept as *.tmp and "
                             "will be moved to raw/_incomplete on the next start")
        print("[ok] python runs complete")
    finally:
        _release_lock()


# ----------------------------------------------------------------------------------------------- C++

def cmd_cxx_pending(args):
    from .manifest import require_cxx_build, require_patch_applied
    from .runs import load_plan, quarantine, select, session_snapshot, state
    require_patch_applied()
    require_cxx_build()
    cdir = _cdir(args)
    plan = load_plan(cdir)
    session_snapshot(cdir, "cxx")
    out = []
    for rec in sorted(select(plan, "cxx", args.study, args.benchmark, args.arm),
                      key=lambda r: (r["spec"]["run_id"], r["key"])):
        st = state(cdir, rec)
        if st == "incomplete":
            quarantine(cdir, rec["key"])
            st = "pending"
        if st == "pending":
            out.append(rec["key"])
    if args.limit:
        out = out[: args.limit]
    for k in out:
        print(k)


def cmd_cxx_prepare(args):
    from .runs import campaign_paths, load_plan, raw_dir, tmp_dir
    cdir = Path(args.campaign_dir).resolve()
    plan = load_plan(cdir)
    if args.key not in plan:
        raise SystemExit(f"[fail] unknown run key {args.key}")
    if raw_dir(cdir, args.key).exists():
        raise SystemExit(f"[fail] {args.key} already published")
    t = tmp_dir(cdir, args.key)
    t.mkdir(parents=True, exist_ok=False)
    shutil.copy2(campaign_paths(cdir)["cxx_conf"] / args.key / "confsys.xcsf", t / "confsys.xcsf")
    print(t)


def cmd_cxx_finalize(args):
    from .cxx import count_coverings, validate_run_dir
    from .manifest import build_info
    from .runs import load_plan, publish, tmp_dir
    cdir = Path(args.campaign_dir).resolve()
    rec = load_plan(cdir)[args.key]
    t = tmp_dir(cdir, args.key)
    try:
        summary = validate_run_dir(t, rec["spec"])
        err = t / "stderr.log.gz"
        summary["n_coverings_logged"] = count_coverings(err) if err.exists() else None
    except Exception as exc:
        raise SystemExit(f"[FAIL] {args.key}: {exc}\n       outputs kept in {t}")
    info = build_info() or {}
    publish(cdir, rec, dict(implementation="cxx", binary_sha256=info.get("binary_sha256"),
                            build_flags=info.get("cxxflags"), validation=summary))
    print(f"[done] {args.key} ({summary['timing_us']/1e6:.1f}s xcslib time)")


# --------------------------------------------------------------------------------------- bookkeeping

def cmd_status(args):
    import collections
    from .runs import load_plan, state
    cdir = _cdir(args)
    plan = load_plan(cdir)
    cnt = collections.Counter()
    for rec in plan.values():
        s = rec["spec"]
        cnt[(s["study"], s["implementation"], state(cdir, rec))] += 1
    print(f"campaign: {cdir}")
    for (study, impl, st), n in sorted(cnt.items()):
        print(f"  {study:8s} {impl:4s} {st:11s} {n}")
    inc = cdir / "raw" / "_incomplete"
    if inc.exists():
        print(f"  quarantined interrupted attempts: {sum(1 for _ in inc.rglob('confsys.xcsf')) + sum(1 for _ in inc.rglob('curve.csv'))}"
              f" (see {inc})")
    return cnt


def cmd_verify(args):
    from .runs import load_plan, state, verify_published
    cdir = _cdir(args)
    plan = load_plan(cdir)
    problems, n_done, n_missing = [], 0, 0
    for rec in plan.values():
        st = state(cdir, rec)
        if st == "done":
            n_done += 1
            problems += verify_published(cdir, rec)
        else:
            n_missing += 1
    for p in problems:
        print("[FAIL]", p)
    print(f"[info] published {n_done}/{len(plan)}; not yet published {n_missing}")
    if problems:
        raise SystemExit(f"[fail] {len(problems)} integrity problem(s)")
    if args.require_complete and n_missing:
        raise SystemExit(f"[fail] campaign incomplete: {n_missing} run(s) missing")
    print("[ok] raw results intact")


# ------------------------------------------------------------------------------------- validation

def cmd_validate(args):
    from . import validation
    out = Path(args.out).resolve() if args.out else CAMPAIGN_DIR / "results" / "_validation"
    validation.run_all(out, with_cxx=not args.no_cxx, with_determinism=not args.no_determinism)


def cmd_collect(args):
    from . import collect
    collect.collect(_cdir(args), allow_incomplete=args.allow_incomplete)


def cmd_analyze(args):
    from . import analysis
    analysis.analyze(_cdir(args))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m xcsfcamp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def campaign_args(p, need_dir=False):
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        p.add_argument("--profile", default=None, help="smoke | pilot | (none = full campaign)")
        p.add_argument("--root", default=None, help="results root (default campaign/results)")
        p.add_argument("--campaign-dir", default=None, help="explicit results/<campaign_id> directory")

    def filters(p):
        p.add_argument("--study")
        p.add_argument("--benchmark")
        p.add_argument("--arm")
        p.add_argument("--limit", type=int, default=0)

    p = sub.add_parser("plan"); campaign_args(p); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("where"); campaign_args(p); p.set_defaults(fn=cmd_where)
    p = sub.add_parser("audit"); campaign_args(p); p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("validate"); p.add_argument("--out"); p.add_argument("--no-cxx", action="store_true")
    p.add_argument("--no-determinism", action="store_true"); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("run-python"); campaign_args(p); filters(p)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--break-lock", action="store_true"); p.set_defaults(fn=cmd_run_python)
    p = sub.add_parser("cxx-pending"); campaign_args(p); filters(p); p.set_defaults(fn=cmd_cxx_pending)
    p = sub.add_parser("cxx-prepare"); p.add_argument("campaign_dir"); p.add_argument("key"); p.set_defaults(fn=cmd_cxx_prepare)
    p = sub.add_parser("cxx-finalize"); p.add_argument("campaign_dir"); p.add_argument("key"); p.set_defaults(fn=cmd_cxx_finalize)
    p = sub.add_parser("status"); campaign_args(p); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("verify"); campaign_args(p); p.add_argument("--require-complete", action="store_true")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("collect"); campaign_args(p); p.add_argument("--allow-incomplete", action="store_true")
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("analyze"); campaign_args(p); p.set_defaults(fn=cmd_analyze)
    args = ap.parse_args(argv)
    from .config import ConfigError
    from .parity import ParityError
    from .runs import RunStateError
    try:
        args.fn(args)
    except (ConfigError, ParityError, RunStateError) as exc:
        raise SystemExit(f"[fail] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
