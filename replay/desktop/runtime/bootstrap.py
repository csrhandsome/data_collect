"""Public interpreter bootstrap; application modules are compiled with Nuitka."""

import multiprocessing
import os
import site
import sys
from pathlib import Path

runtime_root = Path(__file__).resolve().parent
os.environ.pop("PYTHONHOME", None)
os.environ.pop("PYTHONPATH", None)
site.addsitedir(str(runtime_root / "lib/python3.12/site-packages"))
sys.path.insert(0, str(runtime_root))
import desktop_runtime  # noqa: E402

runtime_executable = str(runtime_root / "bin/data-collect-runtime")
os.environ["DATA_COLLECT_EXECUTABLE"] = runtime_executable
multiprocessing.set_executable(runtime_executable)

if __name__ == "__main__":
    if "-c" in sys.argv[1:]:
        # multiprocessing uses the interpreter protocol for spawn/resource_tracker.
        code_index = sys.argv.index("-c") + 1
        code = sys.argv[code_index]
        sys.argv = ["-c", *sys.argv[code_index + 1:]]
        exec(code, {"__name__": "__main__"})
    else:
        raise SystemExit(desktop_runtime.main())
