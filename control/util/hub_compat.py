"""Allow the tested Transformers 4.57.6 / Hub 1.x combination in local ASR loaders."""

import importlib.metadata
import importlib.util
import sys
from pathlib import Path


def allow_hub1_for_transformers() -> None:
    name = "transformers.dependency_versions_table"
    if name in sys.modules or importlib.metadata.version("transformers") != "4.57.6":
        return
    package = importlib.util.find_spec("transformers")
    if package is None or package.origin is None:
        raise ImportError("Transformers is required for ASR")
    spec = importlib.util.spec_from_file_location(
        name, Path(package.origin).with_name("dependency_versions_table.py")
    )
    table = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(table)
    # Preserve all other dependency checks and the minimum Hub version. No model,
    # tokenizer, network API or installed package files are changed.
    if table.deps.get("huggingface-hub") == "huggingface-hub>=0.34.0,<1.0":
        table.deps["huggingface-hub"] = "huggingface-hub>=0.34.0,<2.0"
    sys.modules[name] = table
