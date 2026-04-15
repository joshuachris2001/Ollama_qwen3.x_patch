"""
ModelCores/__init__.py
======================
Plugin auto-discovery for the merge engine.

Scans this directory for .py files (excluding __init__.py and base.py),
imports each module, and collects every class that:
  - subclasses BaseModelCore
  - is not BaseModelCore itself
  - has a non-empty MODEL_TYPE class attribute

The resulting registry maps MODEL_TYPE → class and is consumed by merge.py
to populate the --model-type choices and instantiate the correct plugin.

Adding a new model
------------------
1.  Create ModelCores/<model_name>.py
2.  Define a class that inherits BaseModelCore
3.  Set MODEL_TYPE = "<your_type_string>"
4.  Implement inject_kv() and process_mmproj_tensors()
5.  Done — the engine discovers it automatically on next run.
"""

from __future__ import annotations

import importlib
import inspect
import os

from .base import BaseModelCore


def discover_models() -> dict[str, type[BaseModelCore]]:
    """
    Return a registry dict of {MODEL_TYPE: CoreClass} by scanning this package.
    Import errors in individual plugin files are reported and skipped so a
    broken stub does not prevent other models from loading.
    """
    registry: dict[str, type[BaseModelCore]] = {}
    cores_dir = os.path.dirname(__file__)

    for fname in sorted(os.listdir(cores_dir)):
        # Skip non-Python files, __init__, and base (it has no MODEL_TYPE)
        if not fname.endswith(".py"):
            continue
        if fname in ("__init__.py", "base.py", "qwen_base.py"):
            continue

        module_name = f"{__package__}.{fname[:-3]}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            print(f"  [ModelCores] WARNING: could not import '{module_name}': {exc}")
            continue

        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, BaseModelCore)
                and obj is not BaseModelCore
                and obj.MODEL_TYPE
            ):
                if obj.MODEL_TYPE in registry:
                    existing = registry[obj.MODEL_TYPE].__module__
                    print(
                        f"  [ModelCores] WARNING: duplicate MODEL_TYPE '{obj.MODEL_TYPE}' "
                        f"in {module_name} (already registered from {existing}) — skipping"
                    )
                    continue
                registry[obj.MODEL_TYPE] = obj

    return registry


def load_model_core(
    registry: dict[str, type[BaseModelCore]],
    model_type: str,
) -> BaseModelCore:
    """Instantiate and return the core for the given model_type."""
    cls = registry[model_type]
    return cls(arch=model_type)
