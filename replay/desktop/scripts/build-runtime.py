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
BUILD = REPO / "build/desktop/cpu"
DATA_ANALYSIS_MODULES = ("preprocess_vad", "instruction_audio_window", "dataset_io")


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=REPO, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=BUILD)
    parser.add_argument("--dependency-python", type=Path, default=BUILD / "runtime-venv/bin/python",
                        help="Interpreter of a separate runtime-only dependency environment")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--reuse-dependencies", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    build_root = args.build_dir.resolve()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise SystemExit("Build this release on Linux with Python 3.12")
    build_root.mkdir(parents=True, exist_ok=True)
    sources = [
        *sorted((REPO / "control").rglob("*.py")),
        *sorted((REPO / "replay/backend").rglob("*.py")),
        *sorted((REPO / "replay/scripts").rglob("*.py")),
        REPO / "vr_collect.py",
        *(REPO / "data_analysis" / f"{name}.py" for name in DATA_ANALYSIS_MODULES),
        REPO / "replay/desktop/runtime/desktop_runtime.py",
    ]
    if not args.skip_compile:
        contents = {p.relative_to(REPO): p.read_bytes() for p in sources}
        snapshot = {str(p): hashlib.sha256(body).hexdigest() for p, body in contents.items()}
        source_root = build_root / "sources"
        if source_root.exists():
            shutil.rmtree(source_root)
        for relative, body in contents.items():
            staged = source_root / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(body)
        compiler_env = {**os.environ, "NUITKA_CACHE_DIR": str(build_root / "cache")}
        command = [
            sys.executable, "-m", "nuitka", "--mode=module", "--nofollow-imports",
            "--include-package=control", "--include-package=replay.backend",
            "--include-package=replay.scripts", "--include-module=vr_collect",
            "--include-module=replay", "--include-module=data_analysis",
            *(f"--include-module=data_analysis.{name}" for name in DATA_ANALYSIS_MODULES),
            "--output-dir=" + str(build_root / "nuitka"),
            "--report=" + str(build_root / "compilation-report.xml"),
            "--jobs=" + str(args.jobs),
            str(source_root / "replay/desktop/runtime/desktop_runtime.py"),
        ]
        print("Compiling project Python code with Nuitka", flush=True)
        subprocess.run(command, cwd=source_root, env=compiler_env, check=True)
        (build_root / "build-manifest.json").write_text(json.dumps({
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "sources": snapshot,
        }, indent=2))
    artifacts = [
        path for path in (build_root / "nuitka").glob("desktop_runtime*")
        if any(path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    ]
    if len(artifacts) != 1:
        raise SystemExit("Expected exactly one compiled desktop_runtime extension")
    runtime = build_root / "runtime"
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
    dependency_info = json.loads(subprocess.check_output([
        str(args.dependency_python.absolute()), "-I", "-c",
        "import importlib.metadata, json, sys, sysconfig; print(json.dumps({"
        "'version': list(sys.version_info[:2]), 'site': sysconfig.get_path('purelib'),"
        "'dependencies': {d.metadata['Name']: d.version "
        "for d in importlib.metadata.distributions()}}))",
    ], text=True))
    if dependency_info["version"] != list(sys.version_info[:2]):
        raise SystemExit("Compiler and runtime dependencies must use the same Python version")
    site = Path(dependency_info["site"])
    dependencies = dependency_info["dependencies"]
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
    manifest = json.loads((build_root / "build-manifest.json").read_text())
    manifest["dependencies"] = dict(sorted(dependencies.items()))
    (runtime / "build-manifest.json").write_text(json.dumps(manifest, indent=2))
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
    # Resolve inherited dataset definitions into a portable user template.
    sys.path.insert(0, str(REPO))
    from control.config import load_config

    config = load_config(REPO / "config/train/panda.yaml")
    for key in ("username", "password"):
        config.get("robot", {}).pop(key, None)
    template = build_root / "config/train/panda.yaml"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Runtime ready: {launcher}", flush=True)
    run([launcher, "doctor"])


if __name__ == "__main__":
    main()
