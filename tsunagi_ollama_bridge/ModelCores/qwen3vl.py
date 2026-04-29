"""
ModelCores/qwen3vl.py
=====================
ModelCore plugins for Qwen3-VL (dense) and Qwen3-VL-MoE.

Qwen3-VL specifics
------------------
• REQUIRES_BLOB = False
    All KV values are hardcoded — verified by brute-force comparison against
    the official Ollama blob.  This was the core R&D work of the project;
    Ollama does not ship a Qwen3-VL blob so there is no donor to read from.

• clip.vision.image_size passthrough
    Qwen3-VL uses image_size (not shortest/longest_edge), so the base drop
    of clip.vision.image_size is reverted and a rename is added instead.

• Hardcoded KV fields (inject_kv)
    num_channels, temporal_patch_size, num_positional_embeddings,
    vision.rope.freq_base, vision.shortest_edge (pixel-count semantics),
    mrope_sections, deepstack_visual_indexes, tokenizer fields.

• mmproj pipeline
    Full Qwen pipeline inherited from QwenBaseModelCore:
    QKV split + deepstack renames + patch-embed stacking.

• No LLM tensor renames needed.
• No post-write hook needed.

Qwen3-VL-MoE differences
-------------------------
The MoE variant (qwen3vlmoe) follows the identical pipeline.  The only
runtime difference is general.parameter_count, which is read from the LLM
GGUF when present so finetuned MoE models carry the correct value.
"""

from __future__ import annotations
from ctypes import Array
from typing import override

from gguf import GGUFWriter

from .base import _read_scalar
from .qwen_base import QwenBaseModelCore

def get_deepstack_array(mmproj) -> list[int]:
    return sorted({ int(t.split(".")[2]) for t in mmproj if t.startswith("v.deepstack.") and t.split(".")[2].isdigit() } )


# ---------------------------------------------------------------------------
# Qwen3-VL Dense
# ---------------------------------------------------------------------------

class Qwen3VLModelCore(QwenBaseModelCore):
    """Merge plugin for Qwen3-VL dense models (e.g. 7B)."""

    MODEL_TYPE    = "qwen3vl"
    REQUIRES_BLOB = False
    STATUS        = "stable"

    @classmethod
    def get_help_info(cls) -> dict:  # pyright: ignore[reportMissingTypeArgument]
        return {
            "description":   "Qwen3-VL dense (e.g. 7B) — QKV split + deepstack, no blob required",
            "requires_blob": False,
            "status":        "stable",
            "extra_options": [],
        }

    # ------------------------------------------------------------------
    # KV Drop Set
    # ------------------------------------------------------------------

    def get_kv_drop(self) -> set[str]:
        # Revert the base drop of clip.vision.image_size — Qwen3-VL passes
        # it through renamed to {arch}.vision.image_size (see get_kv_renames).
        # general.parameter_count is dropped here and re-injected below so
        # the output always carries the correct value regardless of whether
        # the finetuned LLM includes it.
        return (super().get_kv_drop() | {
            "general.parameter_count",
        }) - {"clip.vision.image_size"}

    # ------------------------------------------------------------------
    # KV Renames
    # ------------------------------------------------------------------

    def get_kv_renames(self) -> dict[str, str]:
        a = self.arch
        # Add image_size rename — this key was removed from the drop set above
        return super().get_kv_renames() | {
            f"clip.vision.image_size": f"{a}.vision.image_size",
        }

    # ------------------------------------------------------------------
    # KV Injection — All values hardcoded (verified against official blob)
    # ------------------------------------------------------------------

    @override
    def inject_kv(
        self,
        writer: GGUFWriter,
        ref_fields,
        mmproj_fields,
        llm_fields,
        *,
        args,
    ) -> None:
        """
        Inject all architecture-critical KV fields for Qwen3-VL.

        Every value here was determined by brute-force comparison of the
        mmproj + finetuned LLM output against a known-working Ollama model.
        No blob is available for Qwen3-VL so these are the authoritative
        constants for this architecture.
        """
        a = self.arch

        # Vision geometry
        writer.add_uint32(f"{a}.vision.num_channels",              3)
        writer.add_uint32(f"{a}.vision.temporal_patch_size",       2)
        writer.add_uint32(f"{a}.vision.num_positional_embeddings", 2304)
        writer.add_float32(f"{a}.vision.rope.freq_base",           10000.0)
        # shortest_edge uses pixel-count semantics for Jan/Qwen3-VL mmprojs

        writer.add_uint32(f"{a}.vision.longest_edge",              16777216)
        writer.add_uint32(f"{a}.vision.shortest_edge",             65536)

        # RoPE / MRoPE
        writer.add_array(f"{a}.mrope_sections", [24, 20, 20])

        # Deepstack
        print("Writing deepstack ... ", end='')
        writer.add_array(f"{a}.vision.deepstack_visual_indexes", self._deepstack_indices_backup) # writer.add_array(f"{a}.vision.deepstack_visual_indexes", get_deepstack_array(mmproj_fields)) # [8, 16, 24])
        print("done")

        # Tokenizer
        writer.add_bool( "tokenizer.ggml.add_eos_token",     False)
        writer.add_bool( "tokenizer.ggml.add_padding_token", False)
        writer.add_array("tokenizer.ggml.eos_token_ids",     [151645, 151643])

        # writer.add_array(f"{a}.vision.deepstack_visual_indexes", self._deepstack_indices_backup) # see! useful here for a quick patch.

        # Parameter count — prefer LLM's own value so finetuned variants
        # carry the correct count; fall back to the known 7B dense value.
        """
        param_count = (
            int(_read_scalar(llm_fields, "general.parameter_count"))  # pyright: ignore[reportArgumentType]
            if "general.parameter_count" in llm_fields
            else 8_767_123_696
        )
        writer.add_uint64("general.parameter_count", param_count)
        """
        if "general.parameter_count" in llm_fields:
            writer.add_uint64("general.parameter_count", int(_read_scalar(llm_fields, "general.parameter_count")))  # pyright: ignore[reportArgumentType, reportAny]
            self._param_count_written = True # unlikly chance

        # File type — must reflect the actual LLM tensor format
        _ft = (
            int(_read_scalar(llm_fields, "general.file_type"))  # pyright: ignore[reportArgumentType]
            if "general.file_type" in llm_fields
            else 32
        )
        writer.add_uint32("general.file_type", _ft)


    @override
    def post_write_tensors(self, writer, ref, args) -> None:
        """
        Goal is to just assume a parameter_count; unless we want to assume a wrong constant...
        """
        if not self._param_count_written:
            total = self._llm_param_count + self._mmproj_param_count
            writer.add_uint64("general.parameter_count", total)


# ---------------------------------------------------------------------------
# Qwen3-VL MoE
# ---------------------------------------------------------------------------

class Qwen3VLMoEModelCore(Qwen3VLModelCore):
    """
    Merge plugin for Qwen3-VL-MoE (e.g. Jan-v2-VL-max).

    Identical pipeline to Qwen3VLModelCore.  general.parameter_count is
    always read from the LLM GGUF so the correct MoE parameter count is
    preserved — the 7B dense fallback in the parent's inject_kv is not
    appropriate for MoE models.
    """

    MODEL_TYPE    = "qwen3vlmoe"
    REQUIRES_BLOB = False
    STATUS        = "stable"

    @classmethod
    def get_help_info(cls) -> dict:
        return {
            "description":   "Qwen3-VL-MoE (e.g. Jan-v2-VL-max) — QKV split + deepstack, no blob required",
            "requires_blob": False,
            "status":        "stable",
            "extra_options": [],
        }
