"""Nuitka-compiled entry point for the desktop server and supervised tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def serve(argv):
    parser = argparse.ArgumentParser(description="Local desktop backend")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args(argv)
    os.environ["DATA_COLLECT_CONFIG"] = str(args.config.resolve())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.work_dir)

    import uvicorn

    from control.config import load_config
    from replay.backend.main import create_app
    from replay.backend.operations import OperationManager

    config = load_config(args.config)
    root = args.data_root or Path(config.get("replay", {}).get("data_root", "data/dataset"))
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manager = OperationManager(
        args.work_dir,
        runtime_command=[os.environ["DATA_COLLECT_EXECUTABLE"]],
        config_path=args.config.resolve(),
        simulate=args.simulate,
        max_steps=args.max_steps,
    )
    application = create_app(root, manager, args.frontend, shutdown_timeout=None)

    class DesktopServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            if self.started:
                port = self.servers[0].sockets[0].getsockname()[1]
                print(json.dumps({"event": "ready", "port": port}), flush=True)

    DesktopServer(
        uvicorn.Config(application, host="127.0.0.1", port=0, log_level="info")
    ).run()
    return 0


def doctor():
    """Check imports, binary resources and spawned workers without opening devices."""
    import importlib
    import multiprocessing as mp

    modules = [
        "numpy", "pyarrow", "av", "torch", "lerobot.datasets.lerobot_dataset",
        "pyrealsense2", "panda_py", "sounddevice", "teleop_xr", "mink",
        "data_analysis.preprocess_vad", "control.microphone_connector", "control.vr_input",
    ]
    for name in modules:
        importlib.import_module(name)
        print(f"IMPORT OK {name}", flush=True)
    from control.ik_solver.mink_ik_solver import MinkFrankaJointIKSolver

    solver = MinkFrankaJointIKSolver()
    solver.forward_kinematics([0, -0.5, 0, -2, 0, 1.5, 0.7])
    from data_analysis.preprocess_vad import _find_silero_jit_model_path

    assert _find_silero_jit_model_path().is_file()
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    worker = ctx.Process(target=spawn_probe, args=(sender,))
    worker.start()
    sender.close()
    try:
        assert receiver.poll(30), "Spawned worker did not respond"
        assert receiver.recv() == "compiled-worker-ok"
        worker.join(10)
        assert worker.exitcode == 0
    finally:
        receiver.close()
        if worker.is_alive():
            worker.terminate()
            worker.join()
    print("DOCTOR OK: native imports, IK assets, VAD model, compiled spawn worker", flush=True)
    return 0


def spawn_probe(sender):
    from control.util.pose import normalize_quat_xyzw

    normalize_quat_xyzw([0, 0, 0, 1])
    sender.send("compiled-worker-ok")
    sender.close()


def main(argv=None):
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("Expected server, collect, replay, delete, doctor or generate-demo")
    command, rest = args[0], args[1:]
    if command == "server":
        return serve(rest)
    if command == "collect":
        from vr_collect import main as collect

        collect(rest)
        return 0
    if command == "replay":
        from replay.scripts.replay_robot import main as replay

        replay(rest)
        return 0
    if command == "delete":
        from control.collection.deletion import cli

        return cli(rest)
    if command == "doctor":
        return doctor()
    if command == "generate-demo":
        from replay.scripts.generate_demo import generate_demo
        from replay.scripts.generate_image_demo import generate_image_demo

        root = Path(rest[0]).resolve()
        generate_demo(root)
        generate_image_demo(root / "demo_z_images")
        return 0
    raise SystemExit(f"Unknown runtime command: {command}")
