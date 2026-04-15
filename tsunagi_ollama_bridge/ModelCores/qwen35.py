"""
ModelCores/qwen35.py
====================
ModelCore plugins for Qwen3.5-VL dense+SSM hybrid and Qwen3.5-VL-MoE.

Qwen3.5-VL specifics
---------------------
• REQUIRES_BLOB = True
    Architecture-critical values vary between model sizes (edge limits,
    per-layer head counts, MRoPE sections, deepstack indexes) so they are
    always sourced from the official Ollama blob rather than hardcoded.

• SSM hybrid layer rename (get_llm_renames)
    The finetuned LLM stores the SSM delta-time bias as "blk.N.ssm_dt.bias"
    but Ollama's llama.cpp backend expects "blk.N.ssm_dt" (no .bias suffix).
    Layer count is determined from the blob's attention.head_count_kv array.

• rope.mrope_interleaved / ssm.v_head_reordered flags
    Both are injected as True — required for correct position encoding and
    SSM v-head ordering in Ollama's runtime.

• Deepstack indexes sourced from blob
    Qwen3.5 dense models may carry deepstack; MoE models do not.  The blob
    is the authoritative source so inject_kv reads it conditionally.

• mmproj pipeline
    Full Qwen pipeline inherited from QwenBaseModelCore:
    QKV split + deepstack renames (indexes scanned from mmproj tensors)
    + patch-embed stacking.
"""

from __future__ import annotations

from gguf import GGUFWriter

from .base import _read_array, _read_scalar
from .qwen_base import QwenBaseModelCore  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Qwen3.5-VL Dense
# ---------------------------------------------------------------------------

class Qwen35ModelCore(QwenBaseModelCore):
    """Merge plugin for Qwen3.5-VL dense models (e.g. 4B, 7B)."""

    MODEL_TYPE    = "qwen35"
    REQUIRES_BLOB = True
    STATUS        = "stable"

    @classmethod
    def get_help_info(cls) -> dict:  # pyright: ignore[reportMissingTypeArgument]
        return {
            "description":   "Qwen3.5-VL dense + SSM hybrid (e.g. 4B, 7B) — blob required",
            "requires_blob": True,
            "status":        "stable",
            "extra_options": [],
        }

    # ------------------------------------------------------------------
    # KV Drop Set
    # ------------------------------------------------------------------

    def get_kv_drop(self) -> set[str]:
        # QwenBaseModelCore already drops: add_eos_token, add_padding_token,
        # eos_token_ids.  Add Qwen3.5-specific keys re-injected from blob.
        a = self.arch
        return super().get_kv_drop() | {  # pyright: ignore[reportUnknownVariableType]
            f"{a}.attention.head_count_kv",   # per-layer array — re-injected from blob
            f"{a}.image_token_id",             # re-injected from blob
            f"{a}.vision_start_token_id",      # re-injected from blob
            f"{a}.vision_end_token_id",        # re-injected from blob
            f"{a}.vision.longest_edge",        # re-injected from blob
            f"{a}.vision.shortest_edge",       # re-injected from blob
            f"{a}.mrope_sections",             # re-injected from blob
            f"{a}.rope.dimension_sections",    # re-injected from blob
            f"{a}.rope.mrope_section",         # re-injected from blob
            "tokenizer.ggml.padding_token_id", # re-injected below
            "general.parameter_count",         # re-injected from blob
        }

    # ------------------------------------------------------------------
    # KV Injection — Most values sourced from blob
    # ------------------------------------------------------------------

    def inject_kv(self, writer: GGUFWriter, ref_fields, mmproj_fields, llm_fields, *,
        args ) -> None:
        a   = self.arch
        ref = ref_fields

        # Vision geometry
        writer.add_uint32(f"{a}.vision.num_channels",              3)
        writer.add_uint32(f"{a}.vision.temporal_patch_size",       2)
        writer.add_uint32(f"{a}.vision.num_positional_embeddings", 2304)
        writer.add_float32(f"{a}.vision.rope.freq_base",           10000.0)
        # Edge limits vary by model size — always read from blob
        writer.add_uint32(f"{a}.vision.longest_edge",
                          int(_read_scalar(ref, f"{a}.vision.longest_edge")))
        writer.add_uint32(f"{a}.vision.shortest_edge",
                          int(_read_scalar(ref, f"{a}.vision.shortest_edge")))

        # Token IDs (blob) — must match the tokenizer in the merged file
        writer.add_uint32(f"{a}.image_token_id",
                          int(_read_scalar(ref, f"{a}.image_token_id")))
        writer.add_uint32(f"{a}.vision_start_token_id",
                          int(_read_scalar(ref, f"{a}.vision_start_token_id")))
        writer.add_uint32(f"{a}.vision_end_token_id",
                          int(_read_scalar(ref, f"{a}.vision_end_token_id")))

        # RoPE / MRoPE (blob) — wrong values cause garbled position encoding
        mrope   = _read_array(ref, f"{a}.mrope_sections")
        dim_sec = _read_array(ref, f"{a}.rope.dimension_sections")
        mrope_s = (
            _read_array(ref, f"{a}.rope.mrope_section")
            if f"{a}.rope.mrope_section" in ref
            else mrope
        )
        writer.add_array(f"{a}.mrope_sections",          mrope)
        writer.add_array(f"{a}.rope.dimension_sections", dim_sec)
        writer.add_array(f"{a}.rope.mrope_section",      mrope_s)
        writer.add_bool( f"{a}.rope.mrope_interleaved",  True)

        # SSM hybrid flag — v-head ordering has already been applied
        writer.add_bool(f"{a}.ssm.v_head_reordered", True)

        # Attention head counts (blob, per-layer array)
        kv_heads = _read_array(ref, f"{a}.attention.head_count_kv")
        writer.add_array(f"{a}.attention.head_count_kv", kv_heads)

        # Tokenizer settings
        writer.add_uint32("tokenizer.ggml.padding_token_id", 248044)
        writer.add_bool( "tokenizer.ggml.add_eos_token",     False)
        writer.add_bool( "tokenizer.ggml.add_padding_token", False)
        eos_ids = _read_array(ref, "tokenizer.ggml.eos_token_ids")
        writer.add_array("tokenizer.ggml.eos_token_ids", [int(x) for x in eos_ids])

        # SPM probability scores — transplanted from blob to prevent subtle
        # token sampling issues when the finetuned LLM drops this field
        if "tokenizer.ggml.scores" in ref:
            scores = _read_array(ref, "tokenizer.ggml.scores")
            writer.add_array("tokenizer.ggml.scores", scores)

        # Deepstack indexes (blob) — dense models may have them; MoE does not
        if f"{a}.vision.deepstack_visual_indexes" in ref:
            f_ds = ref[f"{a}.vision.deepstack_visual_indexes"]
            if len(f_ds.data) > 0:
                ds_viz = _read_array(ref, f"{a}.vision.deepstack_visual_indexes")
                writer.add_array(f"{a}.vision.deepstack_visual_indexes",
                                 [int(x) for x in ds_viz])
            else:
                writer.add_array(f"{a}.vision.deepstack_visual_indexes", [])
        else:
            writer.add_array(f"{a}.vision.deepstack_visual_indexes", [])

        # General metadata
        writer.add_uint64("general.parameter_count",
                          int(_read_scalar(ref, "general.parameter_count")))
        _ft = (
            int(_read_scalar(llm_fields, "general.file_type"))
            if "general.file_type" in llm_fields
            else 32
        )
        writer.add_uint32("general.file_type", _ft)

    # ------------------------------------------------------------------
    # LLM Tensor Renames — SSM dt bias suffix fix
    # ------------------------------------------------------------------

    def get_llm_renames(self, ref_fields=None) -> dict[str, str]:
        """
        Qwen3.5 finetuned LLMs store the SSM delta-time bias as
        "blk.N.ssm_dt.bias"; Ollama's backend expects "blk.N.ssm_dt".
        Layer count is read from the blob's per-layer head_count_kv array.
        """
        if ref_fields is None:
            return {}
        a = self.arch
        num_layers = len(_read_array(ref_fields, f"{a}.attention.head_count_kv"))
        return {f"blk.{i}.ssm_dt.bias": f"blk.{i}.ssm_dt"
                for i in range(num_layers)}


# ---------------------------------------------------------------------------
# Qwen3.5-VL MoE
# ---------------------------------------------------------------------------

class Qwen35MoEModelCore(Qwen35ModelCore):
    """
    Merge plugin for Qwen3.5-VL-MoE (e.g. 35B-A3B).

    Identical pipeline to Qwen35ModelCore.  The blob will not carry
    deepstack_visual_indexes for MoE models (inject_kv writes an empty
    list in that case), and the mmproj tensor scan in process_mmproj_tensors
    will find no v.deepstack.* tensors, so no deepstack renames are built.
    """

    MODEL_TYPE = "qwen35moe"
    STATUS     = "experimental"

    @classmethod
    def get_help_info(cls) -> dict:  # pyright: ignore[reportMissingTypeArgument]
        return {
            "description":   "Qwen3.5-VL-MoE (e.g. 35B-A3B) — blob required",
            "requires_blob": True,
            "status":        "experimental",
            "extra_options": [],
        }
