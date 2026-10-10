"""Planning, resume and publication of runs (shared by the C++ and Python runners).

Layout of results/<campaign_id>/:
  manifest/            config, plan-time snapshot, one snapshot per run session, parity audit,
                       xcsf_python-vs-upstream.diff (the library modification, AGENTS.md rule 3)
  plan/runs.jsonl      every planned run (key, spec, spec_hash)
  plan/cxx/<key>/confsys.xcsf   generated xcslib configurations (inputs, immutable)
  raw/<key>/           published raw results (immutable; DONE.json written last, then the
                       directory is atomically renamed from <key>.tmp)
  raw/_incomplete/     interrupted attempts, moved aside on restart (never deleted)
  derived/             everything regenerated from raw/ (safe to delete)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import manifest as M
from .config import RunSpec, canonical_json, expand_runs, sha256_json


class RunStateError(RuntimeError):
    pass


def campaign_paths(cdir: Path) -> Dict[str, Path]:
    return dict(manifest=cdir / "manifest", plan=cdir / "plan", runs=cdir / "plan" / "runs.jsonl",
                cxx_conf=cdir / "plan" / "cxx", raw=cdir / "raw", incomplete=cdir / "raw" / "_incomplete",
                derived=cdir / "derived", validation=cdir / "validation")


def write_plan(cfg: dict, cdir: Path) -> Dict[str, int]:
    from .parity import render_confsys
    P = campaign_paths(cdir)
    runs = expand_runs(cfg)
    new = {r.key: r for r in runs}
    existing = load_plan(cdir) if P["runs"].exists() else {}
    changed = [k for k in existing if k in new and existing[k]["spec_hash"] != new[k].spec_hash]
    if changed:
        raise RunStateError(f"{len(changed)} planned run(s) changed specification, e.g. {changed[0]}.\n"
                            "Raw results are immutable: use a new campaign_id for a different configuration.")
    removed = [k for k in existing if k not in new]
    if removed:
        raise RunStateError(f"{len(removed)} previously planned run(s) are no longer in the configuration, "
                            f"e.g. {removed[0]}. Use a new campaign_id.")
    for p in ("manifest", "plan", "raw", "derived"):
        P[p].mkdir(parents=True, exist_ok=True)
    added = 0
    with open(P["runs"], "a") as fh:
        for k, r in new.items():
            if k in existing:
                continue
            fh.write(canonical_json(dict(key=k, spec_hash=r.spec_hash, spec=r.data)) + "\n")
            added += 1
            if r["implementation"] == "cxx":
                d = P["cxx_conf"] / k
                d.mkdir(parents=True, exist_ok=True)
                (d / "confsys.xcsf").write_text(render_confsys(r.data, r.spec_hash))
    cfg_path = P["manifest"] / "config.resolved.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old != cfg:
            (P["manifest"] / f"config.resolved.{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(cfg, indent=2))
    cfg_path.write_text(json.dumps(cfg, indent=2))
    snap = P["manifest"] / "plan_snapshot.json"
    if not snap.exists():
        snap.write_text(json.dumps(M.snapshot(), indent=2))
        # exact diff of xcsf_python against its upstream import (hash in the snapshot just written)
        (P["manifest"] / M.PY_LIB_DIFF_NAME).write_text(M.python_library_diff())
    return dict(total=len(new), added=added, existing=len(existing))


def load_plan(cdir: Path) -> Dict[str, dict]:
    P = campaign_paths(cdir)
    if not P["runs"].exists():
        raise RunStateError(f"no plan in {cdir}; run the plan step first")
    out = {}
    for line in P["runs"].read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if sha256_json(rec["spec"]) != rec["spec_hash"]:
                raise RunStateError(f"plan corrupted: hash mismatch for {rec['key']}")
            out[rec["key"]] = rec
    return out


def select(plan: Dict[str, dict], implementation: Optional[str] = None, study: Optional[str] = None,
           benchmark: Optional[str] = None, arm: Optional[str] = None) -> List[dict]:
    out = []
    for rec in plan.values():
        s = rec["spec"]
        if implementation and s["implementation"] != implementation:
            continue
        if study and s["study"] != study:
            continue
        if benchmark and s["benchmark"] != benchmark:
            continue
        if arm and s["arm"] != arm:
            continue
        out.append(rec)
    return out


def raw_dir(cdir: Path, key: str) -> Path:
    return campaign_paths(cdir)["raw"] / key


def tmp_dir(cdir: Path, key: str) -> Path:
    return raw_dir(cdir, key).with_name(raw_dir(cdir, key).name + ".tmp")


def state(cdir: Path, rec: dict) -> str:
    d = raw_dir(cdir, rec["key"])
    if d.exists():
        done = d / "DONE.json"
        if not done.exists():
            raise RunStateError(f"{d} exists without DONE.json (published directories must be complete)")
        info = json.loads(done.read_text())
        if info.get("spec_hash") != rec["spec_hash"]:
            raise RunStateError(f"{rec['key']}: published with spec_hash {info.get('spec_hash')}, plan has "
                                f"{rec['spec_hash']}")
        return "done"
    if tmp_dir(cdir, rec["key"]).exists():
        return "incomplete"
    return "pending"


def quarantine(cdir: Path, key: str) -> Optional[Path]:
    t = tmp_dir(cdir, key)
    if not t.exists():
        return None
    dest = campaign_paths(cdir)["incomplete"] / f"{key}.{time.strftime('%Y%m%d-%H%M%S')}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(t), str(dest))
    print(f"[warn] interrupted attempt of {key} moved to {dest}", file=sys.stderr, flush=True)
    return dest


def publish(cdir: Path, rec: dict, extra: dict):
    """Write DONE.json (hashes of every file) in the tmp dir, then atomically rename it."""
    t, d = tmp_dir(cdir, rec["key"]), raw_dir(cdir, rec["key"])
    if d.exists():
        raise RunStateError(f"refusing to overwrite published run {d}")
    files = {p.relative_to(t).as_posix(): M.sha256_file(p) for p in sorted(t.rglob("*")) if p.is_file()}
    done = dict(key=rec["key"], spec_hash=rec["spec_hash"], finished_at=M.now(), files=files, **extra)
    (t / "DONE.json").write_text(json.dumps(done, indent=2, allow_nan=False))
    os.replace(t, d)
    # make published raw results read-only (immutability guard)
    for p in d.rglob("*"):
        if p.is_file():
            p.chmod(0o444)


def verify_published(cdir: Path, rec: dict) -> List[str]:
    d = raw_dir(cdir, rec["key"])
    info = json.loads((d / "DONE.json").read_text())
    problems = []
    for rel, digest in info["files"].items():
        p = d / rel
        if not p.exists():
            problems.append(f"{rec['key']}: missing {rel}")
        elif M.sha256_file(p) != digest:
            problems.append(f"{rec['key']}: {rel} modified after publication")
    return problems


def session_snapshot(cdir: Path, label: str):
    P = campaign_paths(cdir)
    P["manifest"].mkdir(parents=True, exist_ok=True)
    snap = M.snapshot()
    plan_snap = json.loads((P["manifest"] / "plan_snapshot.json").read_text())
    for lib in ("xcslib", "xcsf_python", "execution_code"):
        if snap[lib]["sha256"] != plan_snap[lib]["sha256"]:
            what = "campaign execution code (xcsfcamp: " + ", ".join(M.EXECUTION_MODULES) + ")" \
                if lib == "execution_code" else f"{lib} source tree"
            raise SystemExit(f"[fail] {what} changed since the plan was created "
                             f"({plan_snap[lib]['sha256'][:12]} -> {snap[lib]['sha256'][:12]}). "
                             "Restore it or start a new campaign_id (raw results must come from one code version).")
    (P["manifest"] / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{label}-{os.getpid()}.json").write_text(
        json.dumps(snap, indent=2))
    return snap
