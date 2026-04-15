"""
ModelCores/qwen_base.py
=======================
Shared base for all Qwen vision-language model plugins (Qwen3-VL, Qwen3.5-VL).

Provides
--------
QwenBaseModelCore
    Extends BaseModelCore with the shared Qwen KV drop/rename sets and the
    full mmproj tensor pipeline (QKV split, deepstack renames, patch-embed
    stack).  Both Qwen3-VL and Qwen3.5-VL subclass this — each only overrides
    the methods that differ for their architecture.

Module-level helpers (used by QwenBaseModelCore and importable by qwen35.py)
    make_qkv_splitters(vit_hidden)
    build_tensor_renames(vit_depth, ds_idxs)
    stack_patch_embed(vision_tensors, mmproj, vit_hidden)
"""

from __future__ import annotations

import re
import numpy as np
from gguf import GGMLQuantizationType

from .base import (
    BaseModelCore,
    FLOAT_TYPES,
    _read_array,
    _read_scalar,
    write_tensor,
)


# ---------------------------------------------------------------------------
# QKV Splitting
# ---------------------------------------------------------------------------

def make_qkv_splitters(vit_hidden: int):
    """
    Return (split_weight, split_bias) closures over vit_hidden.

    The mmproj stores Q, K, V as a single fused tensor:
        weight: [3 * vit_hidden, vit_hidden]
        bias:   [3 * vit_hidden]
    Ollama expects three separate tensors.  BF16/F16 data is viewed as uint16
    before reshape so the raw bytes are preserved exactly.
    """
    def split_weight(tensor):
        data = np.asarray(tensor.data)
        if tensor.tensor_type in (GGMLQuantizationType.BF16, GGMLQuantizationType.F16):
            data = data.view(np.uint16)
        qkv = data.reshape(3, vit_hidden, vit_hidden)
        return qkv[0].copy(), qkv[1].copy(), qkv[2].copy()

    def split_bias(bias_data, tensor_type=None):
        data = np.asarray(bias_data)
        qkv = data.reshape(3, vit_hidden)
        return qkv[0].copy(), qkv[1].copy(), qkv[2].copy()

    return split_weight, split_bias


# ---------------------------------------------------------------------------
# Tensor Rename Map
# ---------------------------------------------------------------------------

def build_tensor_renames(vit_depth: int, ds_idxs: list[int]) -> dict[str, str]:
    """
    Build the mmproj → merged-file tensor name map for Qwen VL models.

    Three rename categories:
    1. Top-level vision module names (patch embed, position embed, merger FC)
    2. Deepstack merger layers — v.deepstack.{vit_idx}.* → v.deepstack_merger.{merger_idx}.*
    3. Per-block layer norms and FFN layers (ln → norm, ffn_up/down → mlp.linear_fc*)
    """
    renames: dict[str, str] = {
        # Patch and position embeddings
        "v.patch_embd.weight":  "v.patch_embed.weight",
        "v.patch_embd.bias":    "v.patch_embed.bias",
        "v.position_embd.weight": "v.pos_embed.weight",
        # Two-layer MLP projector (mm.0 / mm.2 → merger.linear_fc1 / fc2)
        "mm.0.weight": "v.merger.linear_fc1.weight",
        "mm.0.bias":   "v.merger.linear_fc1.bias",
        "mm.2.weight": "v.merger.linear_fc2.weight",
        "mm.2.bias":   "v.merger.linear_fc2.bias",
        # Post-layer-norm on the merger output
        "v.post_ln.weight": "v.merger.norm.weight",
        "v.post_ln.bias":   "v.merger.norm.bias",
    }

    # Deepstack merger layers — sequential merger indices (0, 1, 2) map to VIT block indices
    for merger_idx, vit_idx in enumerate(ds_idxs):
        for suffix in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                       "norm.weight", "norm.bias"):
            src_field, src_attr = suffix.split(".")
            src = f"v.deepstack.{vit_idx}.{suffix}"
            if src_field in ("fc1", "fc2"):
                dst = f"v.deepstack_merger.{merger_idx}.linear_{src_field}.{src_attr}"
            else:
                dst = f"v.deepstack_merger.{merger_idx}.norm.{src_attr}"
            renames[src] = dst

    # Per-block renames: layer norms and FFN layers for all VIT blocks
    for i in range(vit_depth):
        renames[f"v.blk.{i}.ffn_up.bias"]    = f"v.blk.{i}.mlp.linear_fc1.bias"
        renames[f"v.blk.{i}.ffn_up.weight"]  = f"v.blk.{i}.mlp.linear_fc1.weight"
        renames[f"v.blk.{i}.ffn_down.bias"]  = f"v.blk.{i}.mlp.linear_fc2.bias"
        renames[f"v.blk.{i}.ffn_down.weight"]= f"v.blk.{i}.mlp.linear_fc2.weight"
        renames[f"v.blk.{i}.ln1.bias"]       = f"v.blk.{i}.norm1.bias"
        renames[f"v.blk.{i}.ln1.weight"]     = f"v.blk.{i}.norm1.weight"
        renames[f"v.blk.{i}.ln2.bias"]       = f"v.blk.{i}.norm2.bias"
        renames[f"v.blk.{i}.ln2.weight"]     = f"v.blk.{i}.norm2.weight"

    return renames


# ---------------------------------------------------------------------------
# mmproj Tensor Processing
# ---------------------------------------------------------------------------


_DS_PATTERN = re.compile(r"^v\.deepstack\.(\d+)\.")

def resolve_deepstack_indices(mmproj, vit_depth: int) -> list[int]:
    indices = sorted({
        int(m.group(1))
        for t in mmproj.tensors
        if (m := _DS_PATTERN.match(t.name))
    })
    if indices:
        return indices
    return []

def _process_qwen_mmproj_tensors(
    mmproj, renames: dict[str, str], split_weight, split_bias
) -> dict:
    """
    Walk every tensor in the mmproj, apply renames, split fused QKV tensors,
    and return a dict mapping final_name → tensor or (data, dtype, shape) tuple.

    v.patch_embd.weight.1 is dropped here — it is the temporal slice kept
    separately in the mmproj and gets stacked in stack_patch_embed().
    """
    TENSOR_DROP = {"v.patch_embd.weight.1"}

    vision_tensors: dict = {}
    qkv_tensors: dict = {}

    for t in mmproj.tensors:
        if t.name in TENSOR_DROP:
            print(f"  Dropping: {t.name}")
            continue

        if ".attn_qkv." in t.name:
            qkv_tensors[t.name] = t
            continue

        final_name = renames.get(t.name, t.name)

        if final_name in vision_tensors:
            if t.name == final_name:
                print(f"  Replacing legacy with canonical: '{final_name}'")
                vision_tensors[final_name] = t
            else:
                print(f"  Dropping legacy duplicate: '{t.name}'")
        else:
            vision_tensors[final_name] = t

    print(f"\nSplitting {len(qkv_tensors)} QKV tensors...")
    for qkv_name, qkv_tensor in qkv_tensors.items():
        blk_idx = qkv_name.split(".")[2]
        if "bias" in qkv_name:
            q, k, v = split_bias(qkv_tensor.data, qkv_tensor.tensor_type)
        else:
            q, k, v = split_weight(qkv_tensor)
        for suffix, data in (("q", q), ("k", k), ("v", v)):
            component = "bias" if "bias" in qkv_name else "weight"
            vision_tensors[f"v.blk.{blk_idx}.attn_{suffix}.{component}"] = (
                data, qkv_tensor.tensor_type, None
            )

    return vision_tensors


def stack_patch_embed(vision_tensors: dict, mmproj, vit_hidden: int) -> None:  # pyright: ignore[reportMissingTypeArgument]
    """
    Combine the two temporal patch-embedding kernels into the single tensor
    Ollama's loader expects.

    The mmproj stores them as two separate [vit_hidden, 3, 16, 16] tensors:
        v.patch_embd.weight   — frame 0 (renamed to v.patch_embed.weight)
        v.patch_embd.weight.1 — frame 1 (dropped during tensor processing)

    The combined weight must have shape [vit_hidden*3, 2, 16, 16], e.g.
    [3456, 2, 16, 16] for vit_hidden=1152, matching the official converter.
    """
    t0 = vision_tensors.pop("v.patch_embed.weight")
    t1 = next(t for t in mmproj.tensors if t.name == "v.patch_embd.weight.1")

    d0 = np.asarray(t0.data if hasattr(t0, "data") else t0[0])
    d1 = np.asarray(t1.data)

    patch_out = vit_hidden * 3
    combined = np.stack([d0, d1], axis=2).reshape(patch_out, 2, 16, 16).astype(np.float16)
    vision_tensors["v.patch_embed.weight"] = (combined, GGMLQuantizationType.F16, None)

    print(f"  Stacked patch_embed: {list(d0.shape)} × 2 → {list(combined.shape)}")


# ---------------------------------------------------------------------------
# QwenBaseModelCore
# ---------------------------------------------------------------------------

class QwenBaseModelCore(BaseModelCore):  # pyright: ignore[reportImplicitAbstractClass]
    """
    Shared base for all Qwen VL model plugins.

    Handles the mmproj tensor pipeline that is identical across all Qwen
    variants: VIT dimension discovery, deepstack index resolution, QKV
    splitting, tensor renaming, and patch-embed stacking.

    Subclasses override:
        get_kv_drop()     — add/remove architecture-specific keys
        get_kv_renames()  — add architecture-specific renames
        inject_kv()       — write architecture-critical KV fields
        get_llm_renames() — tensor renames specific to that variant
    """

    # ------------------------------------------------------------------
    # KV Drop Set — Qwen-shared additions to base
    # ------------------------------------------------------------------

    def get_kv_drop(self) -> set[str]:
        return super().get_kv_drop() | {
            "tokenizer.ggml.add_eos_token",
            "tokenizer.ggml.add_padding_token",
            "tokenizer.ggml.eos_token_ids",
        }

    # ------------------------------------------------------------------
    # KV Renames — Qwen-shared additions to base
    # ------------------------------------------------------------------

    def get_kv_renames(self) -> dict[str, str]:
        a = self.arch
        return super().get_kv_renames() | {
            f"clip.vision.spatial_merge_size": f"{a}.vision.spatial_merge_size",
        }

    # ------------------------------------------------------------------
    # mmproj Tensor Processing — Full Qwen pipeline
    # ------------------------------------------------------------------

    def process_mmproj_tensors(self, mmproj, args) -> dict:  # pyright: ignore[reportMissingTypeArgument]
        """
        Full Qwen mmproj pipeline:
        1. Read VIT dimensions from mmproj KV
        2. Resolve deepstack indices (from mmproj KV or evenly spaced fallback)
        3. Build tensor rename map and QKV splitter closures
        4. Process tensors (rename + split fused QKV)
        5. Stack temporal patch embeddings
        """
        mf = mmproj.fields
        vit_hidden = int(_read_scalar(mf, "clip.vision.embedding_length"))
        vit_depth  = int(_read_scalar(mf, "clip.vision.block_count"))
        print(f"  Vision encoder: hidden={vit_hidden}, depth={vit_depth}")

        # Deepstack indices: prefer explicit mmproj KV, fall back to evenly spaced
        ds_idxs = resolve_deepstack_indices(mmproj, vit_depth)  # ← replaces all of it
        if ds_idxs:
            print(f"  Deepstack indices (from mmproj tensors): {ds_idxs}")
            print(f"  Deepstack indices: {ds_idxs}")
        else:
            pass # print("  No deepstack tensors found in mmproj")

        renames = build_tensor_renames(vit_depth, ds_idxs)
        split_weight, split_bias = make_qkv_splitters(vit_hidden)

        print("\nProcessing vision tensors...")
        vision_tensors = _process_qwen_mmproj_tensors(
            mmproj, renames, split_weight, split_bias
        )
        print(f"  Vision tensors after QKV split: {len(vision_tensors)}")

        stack_patch_embed(vision_tensors, mmproj, vit_hidden)
        return vision_tensors

    # ------------------------------------------------------------------
    # LLM Tensor Renames — None needed for base Qwen path
    # ------------------------------------------------------------------

    def get_llm_renames(self, ref_fields=None) -> dict[str, str]:
        return {}
