"""
ModelCores/gemma3.py
=====================
STUB — Gemma 3 model cores for debugging/experimentation.

STATUS: stub — values injected from llm_fields where possible, but many fields
need cross-validation against a real Gemma3 GGUF dump vs Ollama's expectations.

One class is registered here:
  Gemma3ModelCore   → MODEL_TYPE = "gemma3"   (multimodal, vision only)

Reference: ollama/ollama convert/convert_gemma3.go
"""

from __future__ import annotations

import sys
#from typing import override

from gguf import GGUFWriter
import re

from .base import (
    STATUS_EXPERIMENTAL,
    BaseModelCore,
    STATUS_STUB,
    _read_scalar,
    _read_array,
    copy_field,
    write_tensor,
)

# ---------------------------------------------------------------------------
# Gemma 3 — known head counts keyed by block count
# Source: convert_gemma3.go — gemma4BLayerCount / gemma12BLayerCount / gemma27BLayerCount
# ---------------------------------------------------------------------------

_GEMMA3_HEAD_MAP: dict[int, tuple[int, int]] = {
    34: (8,  4),   # 4B
    48: (16, 8),   # 12B
    62: (32, 16),  # 27B
}

# ---------------------------------------------------------------------------
# Vision block tensor rename helper
#
# Authoritative sources:
#   llama.cpp convert_hf_to_gguf.py (Gemma3) — uses shortened names
#   Ollama model/models/gemma3/model_vision.go — Go struct gguf tags
#
# llama.cpp mmproj name      Ollama Go gguf tag       Go struct field
# ─────────────────────────  ───────────────────────  ─────────────────
# v.blk.N.attn_out           v.blk.N.attn_output      AttnOutput
# v.blk.N.ln1                v.blk.N.layer_norm1      LayerNorm1
# v.blk.N.ln2                v.blk.N.layer_norm2      LayerNorm2
# v.blk.N.ffn_up             v.blk.N.mlp.fc1          MLP.FC1
# v.blk.N.ffn_down           v.blk.N.mlp.fc2          MLP.FC2
#
# attn_q / attn_k / attn_v pass through unchanged.
# ---------------------------------------------------------------------------

_GEMMA3_VIS_BLK_RENAMES: dict[str, str] = {
    "attn_out.weight":  "attn_output.weight",
    "attn_out.bias":    "attn_output.bias",
    "ln1.weight":       "layer_norm1.weight",
    "ln1.bias":         "layer_norm1.bias",
    "ln2.weight":       "layer_norm2.weight",
    "ln2.bias":         "layer_norm2.bias",
    "ffn_up.weight":    "mlp.fc1.weight",
    "ffn_up.bias":      "mlp.fc1.bias",
    "ffn_down.weight":  "mlp.fc2.weight",
    "ffn_down.bias":    "mlp.fc2.bias",
}

_GEMMA3_VIS_TOP_RENAMES: dict[str, str] = {
    "v.patch_embd.weight":        "v.patch_embedding.weight",
    "v.patch_embd.bias":          "v.patch_embedding.bias",
    "v.position_embd.weight":     "v.position_embedding.weight",
    "v.post_ln.weight":           "v.post_layernorm.weight",
    "v.post_ln.bias":             "v.post_layernorm.bias",
    "mm.input_projection.weight": "mm.mm_input_projection.weight",
    "mm.soft_emb_norm.weight":    "mm.mm_soft_emb_norm.weight",
}


def _gemma3_vision_rename(name: str) -> str:
    """
    Rename a single mmproj vision tensor from its llama.cpp (ggml-org) name
    to the corresponding Ollama blob name.

    Only v.blk.N.* names are touched; all other tensors pass through unchanged.
    Each tensor receives at most ONE rename lookup — no chaining.
    """
    m = re.match(r"(v\.blk\.\d+\.)(.*)", name)
    if m:
        prefix, suffix = m.group(1), m.group(2)
        if suffix in _GEMMA3_VIS_BLK_RENAMES:
            return prefix + _GEMMA3_VIS_BLK_RENAMES[suffix]
        return name
    return _GEMMA3_VIS_TOP_RENAMES.get(name,name)


# ---------------------------------------------------------------------------
# Gemma3ModelCore
# ---------------------------------------------------------------------------

class Gemma3ModelCore(BaseModelCore):
    """
    Stub merge plugin for Gemma 3 vision models.

    KV source map (mirrors convert_gemma3.go — Gemma3ForConditionalGeneration branch)
    ---------------------------------------------------------------------------------
    llm_fields   → block_count, attention.key/value_length (head_dim),
                   attention.sliding_window, attention.sliding_window_pattern,
                   final_logit_softcapping, rope.freq_base, rope.local.freq_base,
                   rope scaling (yarn), embedding_length, feed_forward_length,
                   mm.tokens_per_image, all tokenizer.ggml.*
    mmproj_fields → vision.block_count, vision.embedding_length,
                   vision.feed_forward_length, vision.image_size,
                   vision.patch_size, vision.num_channels,
                   vision.attention.head_count,
                   vision.attention.layer_norm_epsilon
    hardcoded    → vision.projector.scale_factor = 3
                   attention.head_count / head_count_kv (by block_count lookup)
    """

    MODEL_TYPE: str = "gemma3"
    REQUIRES_BLOB: bool = False
    STATUS: str = STATUS_EXPERIMENTAL

    @classmethod
    #@override
    def get_help_info(cls) -> dict:
        return {
            "description": "Gemma 3 vision models (4B / 12B / 27B)",
            "requires_blob": False,
            "status": STATUS_EXPERIMENTAL,
            "extra_options": [],
        }

    # ── KV drop ──────────────────────────────────────────────────────────────

    #@override
    def get_kv_drop(self) -> set[str]:
        a = self.arch
        return super().get_kv_drop() | {
            # Re-injected from llm_fields — drop passthrough copy to prevent duplicates
            f"{a}.attention.head_count",
            f"{a}.attention.head_count_kv",
            f"{a}.attention.key_length",
            f"{a}.attention.value_length",
            f"{a}.attention.sliding_window",
            f"{a}.attention.sliding_window_pattern",
            f"{a}.final_logit_softcapping",
            f"{a}.rope.freq_base",
            f"{a}.rope.local.freq_base",
            f"{a}.rope.scaling.type",
            f"{a}.rope.scaling.factor",
            f"{a}.rope.scaling.original_context_length",
            f"{a}.rope.scaling.extrapolation_factor",
            f"{a}.rope.scaling.beta_fast",
            f"{a}.rope.scaling.beta_slow",
            f"{a}.embedding_length",
            f"{a}.feed_forward_length",
            f"{a}.mm.tokens_per_image",
            # Re-injected from mmproj_fields
            f"{a}.vision.block_count",
            f"{a}.vision.embedding_length",
            f"{a}.vision.feed_forward_length",
            f"{a}.vision.image_size",
            f"{a}.vision.patch_size",
            f"{a}.vision.num_channels",
            f"{a}.vision.attention.head_count",
            f"{a}.vision.attention.layer_norm_epsilon",
            # Hardcoded constant
            f"{a}.vision.projector.scale_factor",
        }

    # ── KV renames ───────────────────────────────────────────────────────────

    #@override
    def get_kv_renames(self) -> dict[str, str]:
        a = self.arch
        # TODO: VERIFY — same clip.vision.* mapping as Gemma4; confirm field names
        # match what Gemma3 mmprojs actually carry vs Gemma4 mmprojs.
        return {
            "clip.vision.block_count":                  f"{a}.vision.block_count",
            "clip.vision.embedding_length":             f"{a}.vision.embedding_length",
            "clip.vision.attention.head_count":         f"{a}.vision.attention.head_count",
            "clip.vision.attention.layer_norm_epsilon": f"{a}.vision.attention.layer_norm_epsilon",
            "clip.vision.patch_size":                   f"{a}.vision.patch_size",
            "clip.vision.image_mean":                   f"{a}.vision.image_mean",
            "clip.vision.image_std":                    f"{a}.vision.image_std",
            "clip.vision.image_size":                   f"{a}.vision.image_size",
        }

    # ── KV injection ─────────────────────────────────────────────────────────

    #@override
    def inject_kv(self, writer: GGUFWriter, ref_fields, mmproj_fields, llm_fields, *, args) -> None:
        a = self.arch

        # -- Head counts (hardcoded by block_count, fallback to llm_fields scalar) --
        block_count = int(_read_scalar(llm_fields, f"{a}.block_count"))
        n_heads, n_kv_heads = _GEMMA3_HEAD_MAP.get(block_count, (None, None))
        if n_heads is None:
            # TODO: VERIFY — unknown size, read directly from llm_fields
            print(f" [gemma3] Unknown block_count {block_count}, reading heads from llm_fields")
            n_heads    = int(_read_scalar(llm_fields, f"{a}.attention.head_count"))
            n_kv_heads = int(_read_scalar(llm_fields, f"{a}.attention.head_count_kv"))
        writer.add_uint32(f"{a}.attention.head_count",    n_heads)
        writer.add_uint32(f"{a}.attention.head_count_kv", n_kv_heads)
        print(f" [gemma3] heads={n_heads} kv_heads={n_kv_heads} (block_count={block_count})")

        # -- Attention geometry (head_dim → key/value length) --
        # TODO: VERIFY — Go uses TextModel.HeadDim with default 256; confirm llm_fields key name
        head_dim_key = f"{a}.attention.key_length"
        if head_dim_key in llm_fields:
            head_dim = int(_read_scalar(llm_fields, head_dim_key))
        else:
            print(f" [gemma3] WARNING: {head_dim_key} not in llm_fields, defaulting to 256")
            head_dim = 256
        writer.add_uint32(f"{a}.attention.key_length",   head_dim)
        writer.add_uint32(f"{a}.attention.value_length", head_dim)

        # -- Sliding window --
        # TODO: VERIFY — field name in finetuned GGUF vs official Ollama conversion
        sw_key = f"{a}.attention.sliding_window"
        if sw_key in llm_fields:
            writer.add_uint32(sw_key, int(_read_scalar(llm_fields, sw_key)))
        else:
            print(f" [gemma3] WARNING: {sw_key} not found in llm_fields")

        # -- Sliding window pattern (bool array) --
        # Source: layer_types (list of strings) in llm_fields, or reconstruct from interval
        # TODO: VERIFY — finetuned GGUFs may not carry layer_types; need to check what's present
        swp_key = f"{a}.attention.sliding_window_pattern"
        if swp_key in llm_fields:
            swp = _read_array(llm_fields, swp_key)
            writer.add_array(swp_key, swp)
            print(f" [gemma3] sliding_window_pattern from llm_fields ({len(swp)} entries)")
        else:
            print(f" [gemma3] WARNING: {swp_key} not in llm_fields — pattern not written")
            print(f" [gemma3] If Ollama crashes, this field may be required.")

        # -- RoPE (dual bases — global + local) --
        # TODO: VERIFY — key names in finetuned GGUF; Go uses rope_theta and rope_local_base_freq
        rope_base_key  = f"{a}.rope.freq_base"
        rope_local_key = f"{a}.rope.local.freq_base"
        rope_base  = float(_read_scalar(llm_fields, rope_base_key))  if rope_base_key  in llm_fields else 1_000_000.0
        rope_local = float(_read_scalar(llm_fields, rope_local_key)) if rope_local_key in llm_fields else 10_000.0
        if rope_base_key  not in llm_fields: print(f" [gemma3] WARNING: {rope_base_key}  not found, using default 1_000_000.0")
        if rope_local_key not in llm_fields: print(f" [gemma3] WARNING: {rope_local_key} not found, using default 10_000.0")
        writer.add_float32(rope_base_key,  rope_base)
        writer.add_float32(rope_local_key, rope_local)

        # -- Optional YaRN rope scaling --
        # TODO: VERIFY — field names; only written if type == "yarn" and factor > 0
        yarn_type_key = f"{a}.rope.scaling.type"
        if yarn_type_key in llm_fields:
            yarn_type = str(_read_scalar(llm_fields, yarn_type_key))
            if yarn_type == "yarn":
                factor_key = f"{a}.rope.scaling.factor"
                factor = float(_read_scalar(llm_fields, factor_key)) if factor_key in llm_fields else 0.0
                if factor > 0:
                    writer.add_string(yarn_type_key, yarn_type)
                    writer.add_float32(factor_key, factor)
                    orig_key      = f"{a}.rope.scaling.original_context_length"
                    extrap_key    = f"{a}.rope.scaling.extrapolation_factor"
                    beta_fast_key = f"{a}.rope.scaling.beta_fast"
                    beta_slow_key = f"{a}.rope.scaling.beta_slow"
                    writer.add_uint32 (orig_key,      int(  _read_scalar(llm_fields, orig_key))      if orig_key      in llm_fields else 0)
                    writer.add_float32(extrap_key,    float(_read_scalar(llm_fields, extrap_key))    if extrap_key    in llm_fields else 1.0)
                    writer.add_float32(beta_fast_key, float(_read_scalar(llm_fields, beta_fast_key)) if beta_fast_key in llm_fields else 64.0)
                    writer.add_float32(beta_slow_key, float(_read_scalar(llm_fields, beta_slow_key)) if beta_slow_key in llm_fields else 1.0)
                    print(f" [gemma3] YaRN rope scaling written (factor={factor})")

        # -- Final logit softcapping (conditional) --
        softcap_key = f"{a}.final_logit_softcapping"
        if softcap_key in llm_fields:
            softcap = float(_read_scalar(llm_fields, softcap_key))
            if softcap > 0:
                writer.add_float32(softcap_key, softcap)

        # -- LLM dimension fields --
        for key in (f"{a}.embedding_length", f"{a}.feed_forward_length"):
            if key in llm_fields:
                writer.add_uint32(key, int(_read_scalar(llm_fields, key)))
            else:
                print(f" [gemma3] WARNING: {key} not found in llm_fields")

        # -- Vision geometry (from mmproj_fields) --
        # TODO: VERIFY — confirm these clip.* keys are present in Gemma3 mmproj GGUFs
        writer.add_uint32(f"{a}.vision.projector.scale_factor", 3)  # hardcoded Go: if nMerge==0 { nMerge=3 }

        for src_key, dst_key, default_val, writer_fn in (
            #("clip.vision.block_count",                  f"{a}.vision.block_count",                  None,  "add_uint32"),
            #("clip.vision.embedding_length",             f"{a}.vision.embedding_length",             None,  "add_uint32"),
            ("clip.vision.feed_forward_length",          f"{a}.vision.feed_forward_length",          None,  "add_uint32"),
            ("clip.vision.image_size",                   f"{a}.vision.image_size",                   None,  "add_uint32"),
            #("clip.vision.patch_size",                   f"{a}.vision.patch_size",                   None,  "add_uint32"),
            ("clip.vision.num_channels",                 f"{a}.vision.num_channels",                 3,     "add_uint32"),
            #("clip.vision.attention.head_count",         f"{a}.vision.attention.head_count",         None,  "add_uint32"),
            #("clip.vision.attention.layer_norm_epsilon", f"{a}.vision.attention.layer_norm_epsilon", 1e-6,  "add_float32"),
        ):
            if src_key in mmproj_fields:
                val = mmproj_fields[src_key].parts[mmproj_fields[src_key].data[0]][0]
                getattr(writer, writer_fn)(dst_key, val)
            elif default_val is not None:
                print(f" [gemma3] WARNING: {src_key} not in mmproj, using default {default_val}")
                getattr(writer, writer_fn)(dst_key, default_val)
            else:
                print(f" [gemma3] WARNING: {src_key} not in mmproj_fields — field not written")

        # -- mm.tokens_per_image (optional) --
        tpi_key = f"{a}.mm.tokens_per_image"
        if tpi_key in llm_fields:
            writer.add_uint32(tpi_key, int(_read_scalar(llm_fields, tpi_key)))  # pyright: ignore[reportAny, reportArgumentType]
            print(f" [gemma3] mm.tokens_per_image written")
        else:
            print(f" [gemma3] NOTE: {tpi_key} not in llm_fields (may be absent in finetuned GGUFs)")

    # ── mmproj tensor processing ──────────────────────────────────────────────

    #@override
    def process_mmproj_tensors(self, mmproj, args) -> dict:  # pyright: ignore[reportMissingTypeArgument]
        """
        Gemma3 mmproj vision tensor passthrough with per-block renames applied.

        llama.cpp (ggml-org) uses shortened vision block names; Ollama's Go runner
        expects the full names as defined in model/models/gemma3/model_vision.go.
        _gemma3_vision_rename() maps the five mismatched suffixes per block:
            attn_out   → attn_output
            ln1        → layer_norm1
            ln2        → layer_norm2
            ffn_up     → mlp.fc1
            ffn_down   → mlp.fc2
        All other tensors (attn_q/k/v, patch embed, mm.*, rope_freqs.*) pass through
        unchanged. No QKV split, no deepstack, no audio.
        """
        encoder_tensors: dict = {}  # pyright: ignore[reportMissingTypeArgument]
        renamed_count = 0

        for t in mmproj.tensors:
            final_name = _gemma3_vision_rename(t.name)
            if final_name != t.name:
                #print(f" [gemma3] tensor rename: {t.name} → {final_name}")
                renamed_count += 1
            encoder_tensors[final_name] = t

        print(f" [gemma3] Total mmproj tensors : {len(encoder_tensors)}")
        if renamed_count:
            print(f" [gemma3] Vision tensors renamed: {renamed_count}")
        return encoder_tensors