"""
ModelCores/base.py
==================
Shared low-level GGUF utilities and the BaseModelCore abstract class.

Every model plugin in ModelCores/ must subclass BaseModelCore and set MODEL_TYPE.
The merge engine (merge.py) discovers plugins automatically and calls these methods
in a defined order — see merge.py for the call sequence.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import OrderedDict

from gguf.gguf_reader import ReaderField
import numpy as np
from gguf import GGUFWriter, GGUFValueType, GGMLQuantizationType

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FLOAT_TYPES: frozenset[GGMLQuantizationType] = frozenset({
    GGMLQuantizationType.F16,
    GGMLQuantizationType.F32,
    GGMLQuantizationType.BF16,
    GGMLQuantizationType.F64,
})

# GGUF header fields managed automatically by GGUFWriter — never copy these.
SKIP_META: frozenset[str] = frozenset({
    "general.architecture",
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
})

# KV keys present in every mmproj that must be suppressed or re-injected.
COMMON_KV_DROP: frozenset[str] = frozenset({
    "tokenizer.chat_template",
    "clip.has_vision_encoder",
    "clip.has_audio_encoder",
    "clip.projector_type",
    "clip.vision.projector_type",
    "clip.use_gelu",
    "clip.vision.feed_forward_length",
    "clip.vision.image_size",
    "clip.vision.is_deepstack_layers",
    "clip.vision.projection_dim",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.bos_token_id",
    "general.name", "general.type", "general.size_label", "general.license",
    "general.tags", "general.languages", "general.base_model.count",
    "general.base_model.0.name", "general.base_model.0.organization",
    "general.base_model.0.repo_url",
    "general.sampling.top_k", "general.sampling.top_p",
    "general.file_type",
    "general.quantization_version",
})

# Plugin status constants — set STATUS on each subclass.
STATUS_STABLE       = "stable"
STATUS_STUB         = "stub"
STATUS_EXPERIMENTAL = "experimental"

# ---------------------------------------------------------------------------
# Module-level GGUF helpers (importable by plugin modules)
# ---------------------------------------------------------------------------

def _read_array(fields: dict, key: str) -> list:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType, reportUnusedFunction]
    """Return a Python list from a GGUF array field."""
    f = fields[key]
    return np.concatenate([f.parts[idx] for idx in f.data]).tolist()  # pyright: ignore[reportAny, reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]


def _read_scalar(fields: OrderedDict[str, ReaderField], key: str):
    """Return a single scalar value from a GGUF scalar field."""
    f = fields[key]
    return f.parts[f.data[0]][0]  # pyright: ignore[reportAny]


def copy_field(writer: GGUFWriter, field: dict, name: str | None = None) -> None:  # pyright: ignore[reportMissingTypeArgument]
    """
    Copy one KV metadata field from a GGUFReader field object into writer.
    Pass name= to write under a different key (used for clip.* → .* renames).
    Silently skips GGUF header fields that writer manages itself.
    """
    if name is None:
        name = field.name
    if name in SKIP_META:
        return

    vtype = field.types[0]
    d = field.data[0]

    if   vtype == GGUFValueType.UINT8:   writer.add_uint8(name,   field.parts[d][0])
    elif vtype == GGUFValueType.INT8:    writer.add_int8(name,    field.parts[d][0])
    elif vtype == GGUFValueType.UINT16:  writer.add_uint16(name,  field.parts[d][0])
    elif vtype == GGUFValueType.INT16:   writer.add_int16(name,   field.parts[d][0])
    elif vtype == GGUFValueType.UINT32:  writer.add_uint32(name,  field.parts[d][0])
    elif vtype == GGUFValueType.INT32:   writer.add_int32(name,   field.parts[d][0])
    elif vtype == GGUFValueType.FLOAT32: writer.add_float32(name, field.parts[d][0])
    elif vtype == GGUFValueType.UINT64:  writer.add_uint64(name,  field.parts[d][0])
    elif vtype == GGUFValueType.INT64:   writer.add_int64(name,   field.parts[d][0])
    elif vtype == GGUFValueType.FLOAT64: writer.add_float64(name, field.parts[d][0])
    elif vtype == GGUFValueType.BOOL:    writer.add_bool(name,    bool(field.parts[d][0]))
    elif vtype == GGUFValueType.STRING:
        writer.add_string(name, bytes(field.parts[d]).decode("utf-8"))
    elif vtype == GGUFValueType.ARRAY:
        if not field.data:
            print(f"  NOTE: skipping empty array '{name}'")
            return
        etype = field.types[1]
        if etype == GGUFValueType.STRING:
            arr = [bytes(field.parts[idx]).decode("utf-8") for idx in field.data]
        else:
            arr = np.concatenate([field.parts[idx] for idx in field.data]).tolist()
        writer.add_array(name, arr)
    else:
        print(f"  WARNING: skipping unknown type {vtype} for '{name}'")


def write_tensor(
    writer: GGUFWriter,
    name: str,
    data,
    dtype: GGMLQuantizationType,
    shape: list[int] | None = None,
) -> None:
    """
    Write one tensor into writer.

    shape must be in numpy axis order (innermost dimension first); GGUFWriter
    reverses it internally to produce the correct GGUF storage order.
    If shape is None and data has a .shape attribute it is read automatically.
    Quantized types are written without a raw shape override because their
    layout is opaque to numpy.
    """
    if shape is None and hasattr(data, "shape"):
        shape = [int(x) for x in data.shape]
    if dtype in FLOAT_TYPES:
        writer.add_tensor(name, data, raw_shape=shape, raw_dtype=dtype)
    else:
        writer.add_tensor(name, data, raw_dtype=dtype)


# ---------------------------------------------------------------------------
# BaseModelCore
# ---------------------------------------------------------------------------

class BaseModelCore(ABC):
    """
    Abstract base for all model-specific merge plugins.

    Subclasses must set MODEL_TYPE and implement inject_kv() and
    process_mmproj_tensors(). All other methods have sensible defaults
    that can be overridden when the model requires different behaviour.

    Engine call order
    -----------------
    1.  cls.add_args(parser)              — extend CLI before parse
    2.  cls.validate_args(args)           — model-specific arg validation
    3.  core.process_mmproj_tensors(...)  — returns encoder tensor dict
    4.  core.get_kv_drop()               — set of KV keys to suppress
    5.  core.get_kv_renames()            — clip.* → arch.* rename map
    6.  core.should_skip_mmproj_kv(...)  — per-field mmproj KV filter
    7.  core.inject_kv(...)              — write controlled KV fields
    8.  core.get_llm_renames(ref_fields) — LLM tensor name fixes
    9.  core.prepare_llm(llm)            — pre-scan LLM for state
    10. core.should_drop_llm_tensor(...) — per-tensor LLM drop filter
    11. [encoder tensors written by engine]
    12. core.post_write_tensors(...)     — post-write hook (blob transplant etc.)
    """

    MODEL_TYPE:    str  = ""
    REQUIRES_BLOB: bool = True
    STATUS:        str  = STATUS_STABLE

    def __init__(self, arch: str) -> None:
        self.arch = arch

    # ------------------------------------------------------------------
    # Plugin self-description (used by merge.py custom help system)
    # ------------------------------------------------------------------

    @classmethod
    def get_help_info(cls) -> dict[str, str | bool | list[tuple[str, str]]]:
        """
        Return a dict describing this plugin for the custom --help display.

        Keys
        ----
        description : str
            One-line human-readable model description shown next to the type name.
        requires_blob : bool
            Whether --blob is mandatory (echoed from REQUIRES_BLOB by default).
        status : str
            One of STATUS_STABLE / STATUS_STUB / STATUS_EXPERIMENTAL.
            Shown as a badge in the model list so users know what's usable.
        extra_options : list[tuple[str, str]]
            (flag, description) pairs for model-specific CLI options.
            Displayed in a "Model-Specific Options" block grouped by model type.

        Override this in your plugin to provide accurate per-model information.
        The base implementation returns safe defaults derived from class attributes.
        """
        return {
            "description":   "",
            "requires_blob": cls.REQUIRES_BLOB,
            "status":        cls.STATUS,
            "extra_options": [],
        }

    # ------------------------------------------------------------------
    # CLI extension
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser) -> None:
        """Override to add model-specific argparse arguments."""
        pass

    @classmethod
    def format_args_summary(cls, args) -> str | None:
        """
        Optional one-line summary of model-specific flags shown at startup.
        Return a string to print, or None to print nothing.
        """
        return None

    # ------------------------------------------------------------------
    # Argument validation
    # ------------------------------------------------------------------

    @classmethod
    def validate_args(cls, args) -> None:
        """
        Override to validate model-specific arguments.
        Call sys.exit() with a descriptive message on failure.
        """

    # ------------------------------------------------------------------
    # KV metadata filtering
    # ------------------------------------------------------------------

    def get_kv_drop(self) -> set[str]:
        """
        Return the set of KV keys to suppress during LLM/mmproj passthrough.
        Default: the shared COMMON_KV_DROP set.
        Override with super() call to extend:

            def get_kv_drop(self):
                return super().get_kv_drop() | {"my.extra.key"}
        """
        return set(COMMON_KV_DROP)

    def get_kv_renames(self) -> dict[str, str]:
        """
        Return a mapping of {source_key: dest_key} applied during KV passthrough.
        Default: standard clip.vision.* → {arch}.vision.* renames.
        Override entirely for architectures that use a different subset.
        """
        a = self.arch
        return {
            f"clip.vision.block_count":                  f"{a}.vision.block_count",
            f"clip.vision.embedding_length":             f"{a}.vision.embedding_length",
            f"clip.vision.attention.head_count":         f"{a}.vision.attention.head_count",
            f"clip.vision.attention.layer_norm_epsilon": f"{a}.vision.attention.layer_norm_epsilon",
            f"clip.vision.patch_size":                   f"{a}.vision.patch_size",
            f"clip.vision.spatial_merge_size":           f"{a}.vision.spatial_merge_size",
            f"clip.vision.image_mean":                   f"{a}.vision.image_mean",
            f"clip.vision.image_std":                    f"{a}.vision.image_std",
        }

    # ------------------------------------------------------------------
    # mmproj KV conditional filter
    # ------------------------------------------------------------------

    def should_skip_mmproj_kv(
        self, field_name: str, renamed_key: str, args
    ) -> bool:
        """
        Return True to suppress a specific mmproj KV field from passthrough.
        Default: pass everything through.
        """
        return False

    # ------------------------------------------------------------------
    # KV injection (required)
    # ------------------------------------------------------------------

    @abstractmethod
    def inject_kv(
        self,
        writer: GGUFWriter,
        ref_fields: dict | None,
        mmproj_fields: dict,
        llm_fields: dict,
        *,
        args,
    ) -> None:
        """Write all architecture-critical KV fields into writer."""

    # ------------------------------------------------------------------
    # Step 8: LLM tensor renames
    # ------------------------------------------------------------------

    def get_llm_renames(self, ref_fields: dict | None = None, llm_fields: dict | None =None) -> dict[str, str]:  # pyright: ignore[reportMissingTypeArgument]
        """Return a {old_name: new_name} dict for LLM tensors. Default: no renames."""
        return {}

    # ------------------------------------------------------------------
    # Step 9: LLM pre-scan hook
    # ------------------------------------------------------------------

    def prepare_llm(self, llm) -> None:
        """Pre-scan the LLM before the tensor write loop. Default: no-op."""

    # ------------------------------------------------------------------
    # Step 10: LLM tensor drop filter
    # ------------------------------------------------------------------

    def should_drop_llm_tensor(self, name: str, *, args, encoder_tensors: dict) -> bool:
        """
        Return True to skip writing an LLM tensor to the output file.
        Default: drop a.* and v.* (they come from the mmproj instead).
        """
        return name.startswith(("a.", "v."))

    # ------------------------------------------------------------------
    # Step 3: mmproj tensor processing (required)
    # ------------------------------------------------------------------

    @abstractmethod
    def process_mmproj_tensors(self, mmproj, args) -> dict:
        """
        Load, rename, and pre-process all tensors from the mmproj.

        Returns a dict mapping final_name → value, where value is either:
        - A raw GGUFReader tensor object  (has .tensor_type, .data, .shape)
        - A (data, dtype, shape) tuple    for manually assembled tensors
        """

    # ------------------------------------------------------------------
    # Step 12: Post-write hook
    # ------------------------------------------------------------------

    def post_write_tensors(self, writer: GGUFWriter, ref, args) -> None:
        """Post-write hook for clamp transplant etc. Default: no-op."""
