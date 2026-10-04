#!/usr/bin/env python3
"""Compile our Python modules and bundle CPython plus the locked third-party environment.

Third-party packages stay in their upstream form for dynamic imports/JIT compatibility.
No project business source or real datasets are copied into the runtime.
"""

import argparse
import hashlib
import importlib.machinery
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
BUILD = REPO / "build/desktop"


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=REPO, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--reuse-dependencies", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise SystemExit("Build this release on Linux with Python 3.12")
    BUILD.mkdir(parents=True, exist_ok=True)
    sources = [
        *sorted((REPO / "control").rglob("*.py")),
        *sorted((REPO / "replay/backend").rglob("*.py")),
        *sorted((REPO / "replay/scripts").rglob("*.py")),
        REPO / "vr_collect.py", REPO / "data_analysis/preprocess_vad.py",
        REPO / "replay/desktop/runtime/desktop_runtime.py",
    ]
    if not args.skip_compile:
        contents = {p.relative_to(REPO): p.read_bytes() for p in sources}
        snapshot = {str(p): hashlib.sha256(body).hexdigest() for p, body in contents.items()}
        source_root = BUILD / "sources"
        if source_root.exists():
            shutil.rmtree(source_root)
        for relative, body in contents.items():
            staged = source_root / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(body)
        compiler_env = {**os.environ, "NUITKA_CACHE_DIR": str(BUILD / "cache")}
        command = [
            sys.executable, "-m", "nuitka", "--mode=module", "--nofollow-imports",
            "--include-package=control", "--include-package=replay.backend",
            "--include-package=replay.scripts", "--include-module=vr_collect",
            "--include-module=replay", "--include-module=data_analysis",
            "--include-module=data_analysis.preprocess_vad",
            "--output-dir=" + str(BUILD / "nuitka"),
            "--report=" + str(BUILD / "compilation-report.xml"),
            "--jobs=" + str(args.jobs),
            str(source_root / "replay/desktop/runtime/desktop_runtime.py"),
        ]
        print("Compiling project Python code with Nuitka", flush=True)
        subprocess.run(command, cwd=source_root, env=compiler_env, check=True)
        (BUILD / "build-manifest.json").write_text(json.dumps({
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "sources": snapshot,
        }, indent=2))
    artifacts = [
        path for path in (BUILD / "nuitka").glob("desktop_runtime*")
        if any(path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    ]
    if len(artifacts) != 1:
        raise SystemExit("Expected exactly one compiled desktop_runtime extension")
    runtime = BUILD / "runtime"
    if runtime.exists() and not args.reuse_dependencies:
        shutil.rmtree(runtime)
    (runtime / "bin").mkdir(parents=True, exist_ok=True)
    (runtime / "lib").mkdir(exist_ok=True)
    shutil.copy2(Path(sys.executable).resolve(), runtime / "bin/python3.12")
    stdlib = Path(sysconfig.get_path("stdlib"))
    shutil.copytree(
        stdlib, runtime / "lib/python3.12",
        ignore=shutil.ignore_patterns("site-packages", "dist-packages", "__pycache__", "test", "tests"),
        dirs_exist_ok=True,
    )
    library_dir = Path(sysconfig.get_config_var("LIBDIR"))
    shared_library = sysconfig.get_config_var("LDLIBRARY")
    if shared_library and (library_dir / shared_library).exists():
        shutil.copy2((library_dir / shared_library).resolve(), runtime / "lib" / shared_library)
        soname = shared_library + ".1.0"
        if soname != shared_library and not (runtime / "lib" / soname).exists():
            (runtime / "lib" / soname).symlink_to(shared_library)
    site = Path(sysconfig.get_path("purelib"))
    target_site = runtime / "lib/python3.12/site-packages"
    print(f"Bundling third-party dependencies from {site}", flush=True)
    if not args.reuse_dependencies or not target_site.exists():
        shutil.copytree(
            site, target_site,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "__editable__*", "data_collect*"),
        )
    for pth in target_site.glob("*.pth"):
        if str(REPO) in pth.read_text():
            pth.unlink()
    for package in ("control", "replay", "data_analysis", "vr_collect.py"):
        path = target_site / package
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    shutil.copy2(artifacts[0], runtime / artifacts[0].name)
    shutil.copy2(BUILD / "build-manifest.json", runtime / "build-manifest.json")
    shutil.copy2(REPO / "replay/desktop/runtime/bootstrap.py", runtime / "bootstrap.py")
    shutil.copytree(REPO / "data/franka_mjcf", runtime / "data/franka_mjcf", dirs_exist_ok=True)
    launcher = runtime / "bin/data-collect-runtime"
    launcher.write_text(
        '#!/bin/sh\nset -eu\n'
        'runtime_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"\n'
        'export LD_LIBRARY_PATH="$runtime_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
        'exec "$runtime_root/bin/python3.12" -I "$runtime_root/bootstrap.py" "$@"\n'
    )
    launcher.chmod(0o755)
    # Publish a template, never a copy of machine credentials.
    config = yaml.safe_load((REPO / "config/panda.yaml").read_text())
    for key in ("username", "password"):
        config.get("robot", {}).pop(key, None)
    (BUILD / "config").mkdir(exist_ok=True)
    (BUILD / "config/panda.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Runtime ready: {launcher}", flush=True)
    run([launcher, "doctor"])


if __name__ == "__main__":
    main()
