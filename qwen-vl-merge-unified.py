#!/usr/bin/env python3
"""
qwen-vl-merge.py
================
Unified GGUF merger for Ollama-compatible Qwen VL model families.

Merges a finetuned text LLM + a separate mmproj vision encoder into a single
GGUF that Ollama can load directly as a vision-language model.

Supported model types
---------------------
  qwen3vl     — Qwen3-VL (dense, e.g. 7B)
  qwen3vlmoe  — Qwen3-VL-MoE (e.g. Jan-v2-VL-max)
  qwen35      — Qwen3.5-VL (dense + SSM hybrid, e.g. 4B / 7B)

Architecture differences handled automatically
----------------------------------------------
  • Namespace:   clip.vision.* keys are renamed to <arch>.vision.*
  • QKV split:   dimension read from mmproj, never hardcoded
  • Deepstack:   indices read from mmproj KV or evenly spaced
  • SSM rename:  blk.N.ssm_dt.bias → blk.N.ssm_dt  (qwen35 only)
  • KV injection: architecture-critical values (mrope, vision edges, token IDs)
                  are read from the official Ollama blob, not hardcoded, so
                  the output is structurally identical to what Ollama validated

Usage
-----
  python qwen-vl-merge.py \\
      --model-type  qwen35 \\
      --blob        /var/lib/ollama/blobs/sha256-81fb60... \\
      --llm         Qwen3.5-4B.Q6_K.gguf \\
      --mmproj      mmproj.gguf \\
      [--output     merged_qwen.gguf]

  --blob is required for qwen35 (many values are read from it dynamically).
  For qwen3vl and qwen3vlmoe all architecture values are hardcoded, so
  --blob is optional; omit it and the chat template is read from the LLM
  GGUF directly (preferred — finetuners often modify it).

  If --output is omitted the file is written as merged_qwen.gguf in the
  current working directory.
"""

import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SUPPORTED_TYPES = ("qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe")

def parse_args():
    p = argparse.ArgumentParser(
        description="Merge a finetuned Qwen VL LLM + mmproj into a single Ollama-compatible GGUF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-type", "-t",
        required=True,
        choices=SUPPORTED_TYPES,
        metavar="TYPE",
        help=f"Architecture to target. One of: {', '.join(SUPPORTED_TYPES)}",
    )
    p.add_argument(
        "--blob", "-b",
        required=False,
        default=None,
        metavar="PATH",
        help=(
            "Path to the official Ollama base-model blob for this architecture. "
            "Architecture-critical KV values (mrope sections, vision edge limits, "
            "token IDs, head counts) are read from this file so the merged output "
            "matches what Ollama has already validated. "
            "Example: /var/lib/ollama/blobs/sha256-81fb60... "  
            "Required for qwen35. Optional for qwen3vl and qwen3vlmoe — "
            "if omitted the chat template is read from the LLM GGUF instead."
        ),
    )
    p.add_argument(
        "--llm", "-l",
        required=True,
        metavar="PATH",
        help=(
            "Path to the finetuned text model GGUF (input A). "
            "This supplies all LLM tensors and, if present, a finetuned chat template."
        ),
    )
    p.add_argument(
        "--mmproj", "-m",
        required=True,
        metavar="PATH",
        help=(
            "Path to the vision encoder GGUF (input B, the mmproj file). "
            "This supplies all vision tensors and clip.vision.* metadata."
        ),
    )
    p.add_argument(
        "--output", "-o",
        default="merged_qwen.gguf",
        metavar="PATH",
        help="Output path for the merged GGUF. Defaults to merged_qwen.gguf in the CWD.",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Low-level GGUF helpers
# ---------------------------------------------------------------------------

FLOAT_TYPES = {
    GGMLQuantizationType.F16,
    GGMLQuantizationType.F32,
    GGMLQuantizationType.BF16,
    GGMLQuantizationType.F64,
}

# GGUF header fields that GGUFWriter manages automatically — never copy these.
SKIP_META = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}


def _read_array(fields, key):
    """Return a Python list from a GGUF array field."""
    f = fields[key]
    return np.concatenate([f.parts[idx] for idx in f.data]).tolist()


def _read_scalar(fields, key):
    """Return a single scalar value from a GGUF scalar field."""
    f = fields[key]
    return f.parts[f.data[0]][0]


def copy_field(writer, field, name=None):
    """
    Copy one KV metadata field from a GGUFReader field object into writer.
    Pass name= to write it under a different key (used for clip.* → <arch>.* renames).
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


def write_tensor(writer, name, data, dtype, shape=None):
    """
    Write one tensor into writer.

    shape must be in numpy axis order (innermost dimension first); GGUFWriter
    reverses it internally to produce the correct GGUF storage order.
    If shape is None and data has a .shape attribute it is read automatically.
    Quantized types (anything not in FLOAT_TYPES) are written without a raw
    shape override because their layout is opaque to numpy.
    """
    if shape is None and hasattr(data, "shape"):
        shape = [int(x) for x in data.shape]
    if dtype in FLOAT_TYPES:
        writer.add_tensor(name, data, raw_shape=shape, raw_dtype=dtype)
    else:
        writer.add_tensor(name, data, raw_dtype=dtype)

# ---------------------------------------------------------------------------
# Vision QKV splitting
# ---------------------------------------------------------------------------

def make_qkv_splitters(vit_hidden):
    """
    Return (split_weight, split_bias) closures captured over vit_hidden.

    The mmproj stores Q, K, V as a single fused tensor of shape
    [3 * vit_hidden, vit_hidden] (weight) or [3 * vit_hidden] (bias).
    Ollama expects them as three separate tensors. BF16/F16 data is
    viewed as uint16 before reshape so the raw bytes are preserved exactly.
    """
    def split_weight(tensor):
        data = np.asarray(tensor.data)
        if tensor.tensor_type in (GGMLQuantizationType.BF16, GGMLQuantizationType.F16):
            data = data.view(np.uint16)
        qkv = data.reshape(3, vit_hidden, vit_hidden)
        return qkv[0].copy(), qkv[1].copy(), qkv[2].copy()

    def split_bias(bias_data, tensor_type=None):
        data = np.asarray(bias_data)
        # F32 bias — direct reshape, no view needed
        qkv = data.reshape(3, vit_hidden)
        return qkv[0].copy(), qkv[1].copy(), qkv[2].copy()

    return split_weight, split_bias

# ---------------------------------------------------------------------------
# Architecture dispatch table
# ---------------------------------------------------------------------------

def build_arch_config(arch, ref_fields, mmproj_fields):
    """
    Return a dict describing everything that differs between architectures:

      arch_name     — string written to general.architecture
      kv_drop       — set of KV keys to suppress during passthrough
      kv_renames    — dict mapping mmproj clip.* keys → <arch>.vision.* keys
      inject_kv()   — callable(writer, ref, mmproj_fields, llm_fields) that writes all
                      manually controlled KV fields into writer
      llm_renames   — dict of per-layer tensor name fixes (may be empty)
    """
    a = arch  # short alias

    # Keys dropped from every architecture — these are either re-injected
    # manually below, belong only to the standalone mmproj format, or would
    # conflict with values already present in the LLM.
    common_kv_drop = {
        "tokenizer.chat_template",          # re-injected below with correct value
        "clip.has_vision_encoder",          # mmproj-only clip flag
        "clip.projector_type",              # mmproj-only clip flag
        "clip.use_gelu",                    # mmproj-only clip flag
        "clip.vision.feed_forward_length",  # not used by llama.cpp VL loader
        "clip.vision.image_size",           # superseded by shortest/longest_edge
        "clip.vision.is_deepstack_layers",  # bool flag, not needed in merged file
        "clip.vision.projection_dim",       # not used by llama.cpp VL loader
        "tokenizer.ggml.add_bos_token",     # re-injected with correct value
        "tokenizer.ggml.bos_token_id",      # re-injected with correct value
        # Provenance / registry metadata from mmproj — not needed at runtime
        "general.name", "general.type", "general.size_label", "general.license",
        "general.tags", "general.languages", "general.base_model.count",
        "general.base_model.0.name", "general.base_model.0.organization",
        "general.base_model.0.repo_url",
        "general.sampling.top_k", "general.sampling.top_p",
        "general.file_type",            # re-injected manually with correct value
        "general.quantization_version", # re-injected from LLM, not blob
    }

    # clip.vision.* → <arch>.vision.* passthrough renames (same keys, new namespace)
    common_kv_renames = {
        f"clip.vision.block_count":                 f"{a}.vision.block_count",
        f"clip.vision.embedding_length":            f"{a}.vision.embedding_length",
        f"clip.vision.attention.head_count":        f"{a}.vision.attention.head_count",
        f"clip.vision.attention.layer_norm_epsilon":f"{a}.vision.attention.layer_norm_epsilon",
        f"clip.vision.patch_size":                  f"{a}.vision.patch_size",
        f"clip.vision.spatial_merge_size":          f"{a}.vision.spatial_merge_size",
        f"clip.vision.image_mean":                  f"{a}.vision.image_mean",
        f"clip.vision.image_std":                   f"{a}.vision.image_std",
    }

    # qwen3vl / qwen3vlmoe also pass through image_size (they do not use
    # shortest/longest_edge from the blob; the mmproj carries the correct value)
    if a in ("qwen3vl", "qwen3vlmoe"):
        common_kv_renames[f"clip.vision.image_size"] = f"{a}.vision.image_size"

    if a in ("qwen35", "qwen35moe"):
        # qwen35 reads edge limits from the blob, so drop them from passthrough
        # and re-inject them in inject_kv().
        extra_drop = {
            f"{a}.attention.head_count_kv",    # re-injected from blob (per-layer array)
            f"{a}.image_token_id",             # re-injected from blob
            f"{a}.vision_start_token_id",      # re-injected from blob
            f"{a}.vision_end_token_id",        # re-injected from blob
            f"{a}.vision.longest_edge",        # re-injected from blob
            f"{a}.vision.shortest_edge",       # re-injected from blob
            f"{a}.mrope_sections",             # re-injected from blob
            f"{a}.rope.dimension_sections",    # re-injected from blob
            f"{a}.rope.mrope_section",         # re-injected from blob
            "tokenizer.ggml.padding_token_id", # re-injected below
            "tokenizer.ggml.add_eos_token",    # re-injected below
            "tokenizer.ggml.add_padding_token",# re-injected below
            "tokenizer.ggml.eos_token_ids",    # re-injected below
            "general.parameter_count",         # re-injected from blob
        }
        kv_drop = common_kv_drop | extra_drop
    else:
        # qwen3vl / qwen3vlmoe: these keys are simply not present in the LLM
        # source, so no need to drop-and-reinject them.
        kv_drop = common_kv_drop | {
            "tokenizer.ggml.add_eos_token",
            "tokenizer.ggml.add_padding_token",
            "tokenizer.ggml.eos_token_ids",
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.bos_token_id",
        }

    # -----------------------------------------------------------------------
    # Per-architecture inject_kv() implementations
    # -----------------------------------------------------------------------

    def inject_kv_qwen3vl(writer, ref, mf, llm_fields):
        """
        Inject KV fields that are absent from both the LLM and mmproj sources,
        or that must be overridden for the merged file to load correctly.
        All hardcoded values here were verified against the official Ollama blob.
        """
        writer.add_uint32(f"{a}.vision.num_channels",           3)
        writer.add_uint32(f"{a}.vision.temporal_patch_size",    2)
        writer.add_uint32(f"{a}.vision.num_positional_embeddings", 2304)
        writer.add_float32(f"{a}.vision.rope.freq_base",        10000.0)
        # Native mmproj edge limit (Jan models use pixel-count semantics)
        writer.add_uint32(f"{a}.vision.shortest_edge",          65536)
        writer.add_array( f"{a}.mrope_sections",                [24, 20, 20])
        writer.add_array( f"{a}.vision.deepstack_visual_indexes",[8, 16, 24])
        writer.add_uint64("general.parameter_count",            8767123696)
        writer.add_bool("tokenizer.ggml.add_eos_token",         False)
        writer.add_bool("tokenizer.ggml.add_padding_token",     False)
        writer.add_array("tokenizer.ggml.eos_token_ids",        [151645, 151643])
        # general.file_type must match the actual LLM tensor format so tools
        # like ollama show and llama-gguf-split report the correct quant type.
        # Read it from the LLM GGUF directly (same fix as qwen35); fall back to
        # 32 (mixed-precision F16 + quantized) only if the field is absent.
        _ft = (int(_read_scalar(llm_fields, "general.file_type"))
               if "general.file_type" in llm_fields else 32)
        writer.add_uint32("general.file_type", _ft)          # llm

    def inject_kv_qwen35(writer, ref, mf, llm_fields):
        """
        Inject KV fields for qwen35. Architecture-critical values (mrope, vision
        edges, token IDs, head counts) are sourced from the official Ollama blob
        so the merged output is structurally identical to what Ollama validated.
        Values derived from the LLM or mmproj are annotated accordingly.
        """
        # --- Vision geometry (blob) ---
        writer.add_uint32(f"{a}.vision.num_channels",               3)
        writer.add_uint32(f"{a}.vision.temporal_patch_size",        2)
        writer.add_uint32(f"{a}.vision.num_positional_embeddings",  2304)
        writer.add_float32(f"{a}.vision.rope.freq_base",            10000.0)
        writer.add_uint32(f"{a}.vision.longest_edge",  int(_read_scalar(ref, f"{a}.vision.longest_edge")))   # blob
        writer.add_uint32(f"{a}.vision.shortest_edge", int(_read_scalar(ref, f"{a}.vision.shortest_edge")))  # blob

        # --- Token IDs (blob) — must match the tokenizer in the merged file ---
        writer.add_uint32(f"{a}.image_token_id",          int(_read_scalar(ref, f"{a}.image_token_id")))          # blob
        writer.add_uint32(f"{a}.vision_start_token_id",   int(_read_scalar(ref, f"{a}.vision_start_token_id")))   # blob
        writer.add_uint32(f"{a}.vision_end_token_id",     int(_read_scalar(ref, f"{a}.vision_end_token_id")))     # blob

        # --- RoPE / MRoPE (blob) — wrong values cause garbled position encoding ---
        mrope   = _read_array(ref, f"{a}.mrope_sections")
        dim_sec = _read_array(ref, f"{a}.rope.dimension_sections")
        mrope_s = _read_array(ref, f"{a}.rope.mrope_section") \
                  if f"{a}.rope.mrope_section" in ref else mrope
        writer.add_array(f"{a}.mrope_sections",         mrope)    # blob
        writer.add_array(f"{a}.rope.dimension_sections",dim_sec)  # blob
        writer.add_array(f"{a}.rope.mrope_section",     mrope_s)  # blob
        writer.add_bool( f"{a}.rope.mrope_interleaved", True)
        # SSM head-reorder flag — Qwen3.5 uses an SSM hybrid layer; this tells
        # the runtime that v-head ordering has already been applied
        writer.add_bool( f"{a}.ssm.v_head_reordered",  True)

        # --- Attention head counts (blob, per-layer array) ---
        kv_heads = _read_array(ref, f"{a}.attention.head_count_kv")
        writer.add_array(f"{a}.attention.head_count_kv", kv_heads)  # blob

        # --- Tokenizer settings ---
        writer.add_uint32("tokenizer.ggml.padding_token_id",  248044)
        writer.add_bool("tokenizer.ggml.add_eos_token",       False)
        writer.add_bool("tokenizer.ggml.add_padding_token",   False)
        writer.add_array("tokenizer.ggml.eos_token_ids",      [151645, 151643])

        # tokenizer.ggml.scores — SPM probability scores. Finetuned models
        # sometimes drop this field; transplanting from the blob prevents subtle
        # token sampling issues from a missing score table.
        if "tokenizer.ggml.scores" in ref:
            scores = _read_array(ref, "tokenizer.ggml.scores")
            writer.add_array("tokenizer.ggml.scores", scores)  # blob

        # --- General metadata ---
        writer.add_uint64("general.parameter_count", int(_read_scalar(ref, "general.parameter_count")))  # blob
        _ft = (int(_read_scalar(llm_fields, "general.file_type"))
               if "general.file_type" in llm_fields else 32)
        writer.add_uint32("general.file_type", _ft)  # llm
        #writer.add_uint32("general.file_type",        int(_read_scalar(ref, "general.file_type")))        # blob

    # Wire up the correct inject function
    if a == "qwen35":
        inject_kv = inject_kv_qwen35
    else:
        inject_kv = inject_kv_qwen3vl

    # -----------------------------------------------------------------------
    # LLM tensor renames
    # -----------------------------------------------------------------------

    # qwen35: Ollama's llama.cpp backend expects the SSM dt bias stored under
    # "blk.N.ssm_dt" (no .bias suffix). The finetuned GGUF writes it as
    # "blk.N.ssm_dt.bias". We determine the layer count from the blob's
    # head_count_kv array length so this works for any model size.
    if a == "qwen35":
        num_layers = len(_read_array(ref_fields, f"{a}.attention.head_count_kv"))
        llm_renames = {f"blk.{i}.ssm_dt.bias": f"blk.{i}.ssm_dt" for i in range(num_layers)}
    else:
        llm_renames = {}

    return {
        "arch_name":   a,
        "kv_drop":     kv_drop,
        "kv_renames":  common_kv_renames,
        "inject_kv":   inject_kv,
        "llm_renames": llm_renames,
    }

# ---------------------------------------------------------------------------
# Vision tensor processing
# ---------------------------------------------------------------------------

def build_tensor_renames(vit_depth, ds_idxs):
    """
    Build the mmproj → merged-file tensor name mapping.

    Three categories of renames:
      1. Top-level vision module names (patch embed, position embed, merger FC layers)
      2. Deepstack merger layers — v.deepstack.N.* → v.deepstack_merger.M.*
         where M is the merger index (0, 1, 2) and N is the VIT block index
      3. Per-block layer norm and FFN renames (ln → norm, ffn_up/down → mlp.linear_fc*)
    """
    renames = {
        # Patch and position embeddings
        "v.patch_embd.weight":   "v.patch_embed.weight",
        "v.patch_embd.bias":     "v.patch_embed.bias",
        "v.position_embd.weight":"v.pos_embed.weight",
        # Two-layer MLP projector (mm.0 / mm.2 → merger.linear_fc1 / fc2)
        "mm.0.weight":           "v.merger.linear_fc1.weight",
        "mm.0.bias":             "v.merger.linear_fc1.bias",
        "mm.2.weight":           "v.merger.linear_fc2.weight",
        "mm.2.bias":             "v.merger.linear_fc2.bias",
        # Post-layer-norm on the merger output
        "v.post_ln.weight":      "v.merger.norm.weight",
        "v.post_ln.bias":        "v.merger.norm.bias",
    }

    # Deepstack merger layers: the mmproj stores them as v.deepstack.{vit_idx}.*
    # but the merged file must use sequential merger indices (0, 1, 2)
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

    # Per-block renames: layer norms (ln1/ln2 → norm1/norm2) and
    # FFN layers (ffn_up/down → mlp.linear_fc1/fc2) for all VIT blocks
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


def process_mmproj_tensors(mmproj, renames, split_weight, split_bias):
    """
    Walk every tensor in the mmproj, apply renames, split fused QKV tensors,
    and return a dict mapping final_name → tensor-or-(data, dtype, shape) tuple.

    Tensors to unconditionally drop:
      v.patch_embd.weight.1 — temporal slice kept separate in the mmproj;
                              it gets stacked with weight.0 in the next step.
    """
    TENSOR_DROP = {"v.patch_embd.weight.1"}

    vision_tensors = {}
    qkv_tensors    = {}

    for t in mmproj.tensors:
        if t.name in TENSOR_DROP:
            print(f"  Dropping: {t.name}")
            continue

        # Fused QKV tensors need splitting — defer them
        if ".attn_qkv." in t.name:
            qkv_tensors[t.name] = t
            continue

        final_name = renames.get(t.name, t.name)

        # Handle duplicate keys: prefer the canonical (renamed) form over any
        # legacy name that maps to the same destination
        if final_name in vision_tensors:
            if t.name == final_name:
                print(f"  Replacing legacy with canonical: '{final_name}'")
                vision_tensors[final_name] = t
            else:
                print(f"  Dropping legacy duplicate: '{t.name}'")
        else:
            vision_tensors[final_name] = t

    # Split fused QKV tensors into separate Q / K / V entries
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


def stack_patch_embed(vision_tensors, mmproj, vit_hidden):
    """
    Combine the two temporal patch-embedding kernels into the single tensor
    Ollama's loader expects.

    The mmproj stores them as two separate tensors of shape [vit_hidden, 3, 16, 16]:
      v.patch_embd.weight   — frame 0  (already renamed to v.patch_embed.weight)
      v.patch_embd.weight.1 — frame 1  (dropped during tensor processing above)

    The combined weight must have shape [vit_hidden*3, 2, 16, 16] = [3456, 2, 16, 16]
    (for vit_hidden=1152), matching what the official converter produces from the
    original PyTorch checkpoint.
    """
    t0 = vision_tensors.pop("v.patch_embed.weight")
    t1 = next(t for t in mmproj.tensors if t.name == "v.patch_embd.weight.1")

    d0 = np.asarray(t0.data)   # [vit_hidden, 3, 16, 16]
    d1 = np.asarray(t1.data)   # [vit_hidden, 3, 16, 16]

    # Stack along the temporal axis, then reshape to merge out_channels and
    # input_channels as the official converter does
    patch_out = vit_hidden * 3
    combined  = np.stack([d0, d1], axis=2).reshape(patch_out, 2, 16, 16).astype(np.float16)
    vision_tensors["v.patch_embed.weight"] = (combined, GGMLQuantizationType.F16, None)

    print(f"  Stacked patch_embed: {list(d0.shape)} × 2 → {list(combined.shape)}")


# ---------------------------------------------------------------------------
# Main merge routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate paths before doing any real work.
    # --blob is required for qwen35 (values are read from it dynamically).
    # For qwen3vl / qwen3vlmoe it is optional; the chat template comes from the LLM.
    if args.model_type == "qwen35" and not args.blob:
        sys.exit("ERROR: --blob is required for model type 'qwen35'")

    for label, path in [
        ("LLM",    args.llm),
        ("mmproj", args.mmproj),
    ]:
        if not os.path.exists(path):
            sys.exit(f"ERROR: {label} path does not exist: {path}")

    if args.blob and not os.path.exists(args.blob):
        sys.exit(f"ERROR: blob path does not exist: {args.blob}")

    print(f"Model type : {args.model_type}")
    print(f"Blob       : {args.blob}")
    print(f"LLM        : {args.llm}")
    print(f"mmproj     : {args.mmproj}")
    print(f"Output     : {args.output}")
    print()

    # ------------------------------------------------------------------
    # 1. Load reference blob (required for qwen35, optional for qwen3vl/qwen3vlmoe)
    # ------------------------------------------------------------------
    if args.blob:
        print("Loading reference blob...")
        ref        = GGUFReader(args.blob)
        ref_fields = ref.fields
        tmpl_field = ref_fields["tokenizer.chat_template"]
        official_chat_template = bytes(tmpl_field.parts[tmpl_field.data[0]]).decode("utf-8")
        print(f"  chat template: {len(official_chat_template)} chars")
    else:
        # No blob provided — qwen3vl / qwen3vlmoe only.
        # All injected values for those architectures are hardcoded, so the blob
        # is not needed. The chat template will be sourced from the LLM below.
        ref        = None
        ref_fields = None
        official_chat_template = None
        print("No blob provided — all KV values will be hardcoded (qwen3vl/qwen3vlmoe).")

    # Build the architecture config (KV maps, inject function, LLM renames).
    # ref_fields may be None here for qwen3vl/qwen3vlmoe without a blob; that is
    # safe because inject_kv_qwen3vl() never reads from it.
    arch_cfg = build_arch_config(args.model_type, ref_fields, None)

    # ------------------------------------------------------------------
    # 2. Load mmproj — read vision encoder dimensions before processing tensors
    # ------------------------------------------------------------------
    print("\nLoading mmproj...")
    mmproj = GGUFReader(args.mmproj)
    mf     = mmproj.fields

    vit_hidden = int(_read_scalar(mf, "clip.vision.embedding_length"))
    vit_depth  = int(_read_scalar(mf, "clip.vision.block_count"))
    print(f"  Vision encoder: hidden={vit_hidden}, depth={vit_depth}")

    # Deepstack indices: read from the mmproj if explicitly stored, otherwise
    # space them evenly at 1/3, 2/3, and 3/3 of vit_depth
    if "clip.vision.is_deepstack_layers" in mf:
        ds_key = next((k for k in mf if "deepstack" in k and "index" in k), None)
        if ds_key:
            ds_idxs = [int(x) for x in _read_array(mf, ds_key)]
        else:
            step    = vit_depth // 3
            ds_idxs = [step, step * 2, step * 3]
    else:
        step    = vit_depth // 3
        ds_idxs = [step, step * 2, step * 3]
    print(f"  Deepstack indices: {ds_idxs}")

    # Build tensor rename map and QKV splitters now that dimensions are known
    renames             = build_tensor_renames(vit_depth, ds_idxs)
    split_weight, split_bias = make_qkv_splitters(vit_hidden)

    # ------------------------------------------------------------------
    # 3. Process vision tensors (rename, QKV split, patch-embed stack)
    # ------------------------------------------------------------------
    print("\nProcessing vision tensors...")
    vision_tensors = process_mmproj_tensors(mmproj, renames, split_weight, split_bias)
    print(f"  Vision tensors after QKV split: {len(vision_tensors)}")

    stack_patch_embed(vision_tensors, mmproj, vit_hidden)

    # ------------------------------------------------------------------
    # 4. Load finetuned LLM
    # ------------------------------------------------------------------
    print("\nLoading LLM...")
    llm = GGUFReader(args.llm)

    # Prefer the finetuned model's own chat template when present — fine-tuners
    # often modify the template to reflect instruction-following changes.
    # Fall back to the blob's template if the LLM has none (and a blob was given).
    # If neither source has a template, abort with a clear error.
    if "tokenizer.chat_template" in llm.fields:
        f = llm.fields["tokenizer.chat_template"]
        chat_template = bytes(f.parts[f.data[0]]).decode("utf-8")
        print(f"  Using LLM chat template ({len(chat_template)} chars)")
    elif official_chat_template is not None:
        chat_template = official_chat_template
        print("  WARNING: LLM has no chat template — falling back to blob template")
    else:
        sys.exit(
            "ERROR: No chat template found. The LLM GGUF does not carry one and "
            "no --blob was provided to fall back to. Re-run with --blob, or use "
            "a model that includes tokenizer.chat_template."
        )

    # Carry the LLM's quantization_version into the output so downstream tools
    # know which quantization scheme was applied to the LLM tensors
    llm_quant_version = (
        int(_read_scalar(llm.fields, "general.quantization_version"))
        if "general.quantization_version" in llm.fields
        else 2
    )

    # ------------------------------------------------------------------
    # 5. Open writer and write all KV metadata
    # ------------------------------------------------------------------
    writer = GGUFWriter(args.output, arch=args.model_type)

    kv_drop    = arch_cfg["kv_drop"]
    kv_renames = arch_cfg["kv_renames"]

    # Pass through LLM KV — skip anything we will re-inject manually
    print("\nCopying LLM KV metadata...")
    for field in llm.fields.values():
        if field.name in kv_drop:
            continue
        copy_field(writer, field, name=kv_renames.get(field.name, field.name))

    # Pass through mmproj KV — skip keys already present from the LLM and
    # anything in the drop list; rename clip.vision.* → <arch>.vision.*
    print("Copying mmproj vision KV metadata...")
    llm_keys = set(llm.fields.keys())
    for field in mmproj.fields.values():
        if field.name in llm_keys or field.name in kv_drop:
            continue
        if field.name in SKIP_META:
            continue
        renamed = kv_renames.get(field.name, field.name)
        if renamed != field.name:
            print(f"  KV rename: {field.name} → {renamed}")
        copy_field(writer, field, name=renamed)

    # Inject architecture-critical KV fields with verified values
    print("Injecting controlled KV fields...")
    arch_cfg["inject_kv"](writer, ref_fields, mf, llm.fields)

    # Re-inject fields that require values from multiple sources
    writer.add_string("tokenizer.chat_template", chat_template)
    print(f"  ✓ chat template ({len(chat_template)} chars)")
    writer.add_uint32("general.quantization_version", llm_quant_version)

    # ------------------------------------------------------------------
    # 6. Write LLM tensors
    # ------------------------------------------------------------------
    llm_renames   = arch_cfg["llm_renames"]
    dropped_tensors = []

    for t in tqdm(llm.tensors, desc="Writing LLM tensors", unit="tensor", leave=True):
        final_name = llm_renames.get(t.name, t.name)
        data = np.asarray(t.data)
        if t.tensor_type == GGMLQuantizationType.BF16:
            # View BF16 as uint16 so numpy doesn't try to interpret the raw bytes
            data = data.view(np.uint16)
        shape = [int(x) for x in t.shape[::-1]] if t.tensor_type in FLOAT_TYPES else None
        write_tensor(writer, final_name, data, t.tensor_type, shape)

    if dropped_tensors:
        print(f"  Dropped {len(dropped_tensors)} LLM tensors: {dropped_tensors[:5]}")

    # ------------------------------------------------------------------
    # 7. Write vision tensors
    # ------------------------------------------------------------------
    for final_name, t_or_tuple in tqdm(
        vision_tensors.items(), desc="Writing vision tensors", unit="tensor", leave=True
    ):
        if hasattr(t_or_tuple, "tensor_type"):
            # Raw GGUFReader tensor object
            t     = t_or_tuple
            data  = np.asarray(t.data)
            dtype = t.tensor_type
            if dtype == GGMLQuantizationType.BF16:
                data = data.view(np.uint16)
            # Derive shape from GGUF-order t.shape (reversed to numpy order)
            shape = [int(x) for x in t.shape[::-1]] if dtype in FLOAT_TYPES else None
        else:
            # Pre-processed tuple from QKV splitting or patch_embed stacking
            data, dtype, shape = t_or_tuple

        # v.pos_embed.weight: the GGUF writer's dimension reversal interacts badly
        # with how this tensor was stored; force it to F16 with explicit numpy shape
        # [seq_len, hidden] so the writer writes GGUF shape [hidden, seq_len].
        if final_name == "v.pos_embed.weight":
            data  = np.asarray(data).astype(np.float16)
            dtype = GGMLQuantizationType.F16
            shape = [2304, vit_hidden]

        write_tensor(writer, final_name, data, dtype, shape)

    # ------------------------------------------------------------------
    # 8. Finalize and close
    # ------------------------------------------------------------------
    print("\nFinalizing output file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    written_llm = len(llm.tensors) - len(dropped_tensors)
    print(f"\nOutput : {args.output}")
    print(f"  LLM tensors    : {written_llm}"
          + (f" ({len(dropped_tensors)} dropped)" if dropped_tensors else ""))
    print(f"  Vision tensors : {len(vision_tensors)}")
    print(f"  Total          : {written_llm + len(vision_tensors)}")
    print("Done.")


if __name__ == "__main__":
    main()
