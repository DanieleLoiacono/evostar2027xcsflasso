"""Experiment manifest: environment, library fingerprints, patch status, build info."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .config import CAMPAIGN_DIR, CXX_LIB_DIR, PROJECT_DIR, PY_LIB_DIR

# xcslib modifications, applied in this order and committed separately (scripts/01_apply_cxx_patch.sh)
PATCH_FILES = (
    CAMPAIGN_DIR / "patches" / "xcslib-benchmark-functions.patch",   # benchmark functions + min/max input keys
    CAMPAIGN_DIR / "patches" / "xcslib-rls-delta.patch",             # prediction::rls_delta (paper's RLS)
)
PATCH_FILE = PATCH_FILES[0]   # backwards compatibility
BUILD_DIR = CAMPAIGN_DIR / "build"
BUILD_INFO = BUILD_DIR / "build_info.json"
CXX_BIN = BUILD_DIR / "bin" / "xcsf-rf"
PF_DRIVER_BIN = BUILD_DIR / "bin" / "pf_driver"

_EXCLUDE_DIRS = {"__pycache__", ".git", "build", "executables", ".vscode", ".pytest_cache", ".mypy_cache"}
_EXCLUDE_SUFFIX = (".pyc", ".pyo", ".o", ".DS_Store")


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(root: Path) -> Dict[str, Any]:
    """Content hash of a source tree (ignores caches, build products and egg-info)."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _EXCLUDE_DIRS and not d.endswith(".egg-info"))
        for f in sorted(filenames):
            if f.endswith(_EXCLUDE_SUFFIX) or f == ".DS_Store":
                continue
            p = Path(dirpath) / f
            files.append((p.relative_to(root).as_posix(), sha256_file(p)))
    h = hashlib.sha256()
    for rel, digest in files:
        h.update(f"{rel}\0{digest}\n".encode())
    return dict(path=str(root), sha256=h.hexdigest(), n_files=len(files))


def _cmd(args, cwd=None) -> str:
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # pragma: no cover - informative only
        return f"<unavailable: {exc}>"


def patches_sha256() -> str:
    """Combined hash of all xcslib patch files (order-sensitive)."""
    h = hashlib.sha256()
    for p in PATCH_FILES:
        h.update(f"{p.name}\0{sha256_file(p)}\n".encode())
    return h.hexdigest()


def _single_patch_status(path: Path) -> str:
    if shutil.which("git") is None:
        return "unknown (git not found)"
    rev = subprocess.run(["git", "apply", "--check", "--reverse", str(path)], cwd=PROJECT_DIR,
                         capture_output=True, text=True)
    fwd = subprocess.run(["git", "apply", "--check", str(path)], cwd=PROJECT_DIR, capture_output=True, text=True)
    return "applied" if rev.returncode == 0 else ("not-applied" if fwd.returncode == 0 else "conflict")


def patch_status() -> Dict[str, Any]:
    """Status of every xcslib patch ('applied' | 'not-applied' | 'conflict', judged with git apply --check,
    no changes made). The overall status is 'applied' only if every patch is applied."""
    patches = [dict(patch=str(p), name=p.name, sha256=sha256_file(p), text=p.read_text(),
                    status=_single_patch_status(p)) for p in PATCH_FILES]
    states = {p["status"] for p in patches}
    overall = "applied" if states == {"applied"} else ("; ".join(f"{p['name']}: {p['status']}" for p in patches))
    out = dict(status=overall, patches=patches, patch_sha256=patches_sha256(),
               patch_text="".join(p["text"] for p in patches))
    if shutil.which("git") is not None:
        out["working_tree_diff_vs_HEAD"] = _cmd(["git", "diff", "--no-color", "HEAD", "--", CXX_LIB_DIR.name,
                                                 PY_LIB_DIR.name], cwd=PROJECT_DIR)
    return out


def require_patch_applied():
    st = patch_status()
    if st["status"] != "applied":
        raise SystemExit(f"[fail] xcslib patch status: {st['status']}. Run scripts/01_apply_cxx_patch.sh first.")


def git_info() -> Dict[str, Any]:
    if shutil.which("git") is None:
        return dict(available=False)
    return dict(head=_cmd(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR),
                branch=_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_DIR),
                status_porcelain=_cmd(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_DIR))


def environment() -> Dict[str, Any]:
    versions = {}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "matplotlib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return dict(
        timestamp=now(), hostname=socket.gethostname(), platform=platform.platform(), machine=platform.machine(),
        cpu_count=os.cpu_count(), python=sys.version, python_executable=sys.executable, packages=versions,
        cxx_compiler=_cmd([os.environ.get("CXX", "c++"), "--version"]).splitlines()[:1],
        gsl=_cmd(["gsl-config", "--version"]) if shutil.which("gsl-config") else None,
        thread_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                    "VECLIB_MAXIMUM_THREADS")},
    )


def build_info() -> Dict[str, Any] | None:
    if not BUILD_INFO.exists():
        return None
    return json.loads(BUILD_INFO.read_text())


def require_pf_driver() -> Path:
    """The predictor-level test driver built together with xcsf-rf (scripts/02_build_cxx.sh)."""
    info = require_cxx_build()
    if not PF_DRIVER_BIN.exists() or "pf_driver_sha256" not in info:
        raise SystemExit("[fail] campaign/build/bin/pf_driver missing: rebuild with scripts/02_build_cxx.sh.")
    if sha256_file(PF_DRIVER_BIN) != info["pf_driver_sha256"]:
        raise SystemExit("[fail] campaign/build/bin/pf_driver does not match build_info.json; rebuild.")
    return PF_DRIVER_BIN


def require_cxx_build() -> Dict[str, Any]:
    info = build_info()
    if info is None or not CXX_BIN.exists():
        raise SystemExit("[fail] C++ binary missing. Run scripts/02_build_cxx.sh first.")
    if sha256_file(CXX_BIN) != info["binary_sha256"]:
        raise SystemExit("[fail] campaign/build/bin/xcsf-rf does not match build_info.json; rebuild.")
    if info.get("patch_sha256") != patches_sha256():
        raise SystemExit("[fail] the xcslib patches changed after the build; rebuild (scripts/02_build_cxx.sh).")
    current = tree_hash(CXX_LIB_DIR)["sha256"]
    if current != info["source_tree_sha256"]:
        raise SystemExit("[fail] xcslib sources changed after the build; rebuild (and start a new campaign id "
                         "if runs were already produced).")
    return info


EXECUTION_MODULES = ("benchmarks.py", "config.py", "parity.py", "python_runner.py", "cxx.py", "runs.py")


def execution_code_hash() -> Dict[str, Any]:
    """Hash of the campaign modules that influence how runs are executed/validated (not the analysis)."""
    h = hashlib.sha256()
    for name in EXECUTION_MODULES:
        p = CAMPAIGN_DIR / "xcsfcamp" / name
        h.update(f"{name}\0{sha256_file(p)}\n".encode())
    return dict(modules=list(EXECUTION_MODULES), sha256=h.hexdigest())


def snapshot() -> Dict[str, Any]:
    return dict(environment=environment(), git=git_info(), xcslib=tree_hash(CXX_LIB_DIR),
                xcsf_python=tree_hash(PY_LIB_DIR), campaign_code=tree_hash(CAMPAIGN_DIR / "xcsfcamp"),
                execution_code=execution_code_hash(),
                patch=patch_status(), cxx_build=build_info())


VALIDATION_DIR = CAMPAIGN_DIR / "results" / "_validation"


def require_validation() -> Dict[str, Any]:
    """Planning gate: a full `validate` (benchmarks, plots, parity, C++ functions, determinism) must have
    passed for the current build, libraries and execution code (BENCHMARK_FUNCTIONS.md, 'Benchmark validation')."""
    stamp_path = VALIDATION_DIR / "validation_stamp.json"
    if not stamp_path.exists():
        raise SystemExit("[fail] no successful full validation found: run scripts/03_validate.sh first "
                         "(reference plots and checks must precede any XCSF experiment)")
    stamp = json.loads(stamp_path.read_text())
    current = dict(binary_sha256=sha256_file(CXX_BIN) if CXX_BIN.exists() else None,
                   xcslib=tree_hash(CXX_LIB_DIR)["sha256"], xcsf_python=tree_hash(PY_LIB_DIR)["sha256"],
                   execution_code=execution_code_hash()["sha256"])
    changed = [k for k, v in current.items() if stamp.get(k) != v]
    if changed:
        raise SystemExit(f"[fail] the last validation does not match the current {', '.join(changed)}: "
                         "rerun scripts/03_validate.sh")
    missing = [p for p in stamp.get("reference_plots", []) if not (VALIDATION_DIR / p).exists()] or \
        ([] if stamp.get("reference_plots") else ["benchmarks_1d.png"])
    if missing:
        raise SystemExit(f"[fail] reference plots missing ({missing}): rerun scripts/03_validate.sh")
    return stamp
