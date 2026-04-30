"""
ModelCores/nemotron3omni.py
===========================
ModelCore plugin for NVIDIA Nemotron 3 Nano Omni (30B-A3B).

Model-specific needs handled here
-----------------------------------
- Arch rename     LLM GGUF arch is "nemotron_h" / "nemotron_h_moe";
                  merged Ollama output arch is "nemotron_h_omni".
                  All LLM KV fields must be re-keyed nemotron_h.* -> nemotron_h_omni.*
- CLI flags       --vision / --audio to select encoder(s)
- KV drop         All nemotron_h.* (re-injected under nemotron_h_omni.*),
                  clip.* mmproj keys sourced from mmproj_fields
- KV renames      clip.vision.* -> nemotron_h_omni.vision.*
- inject_kv       Sources from llm_fields (nemotron_h.*) and mmproj_fields (clip.*);
                  writes under nemotron_h_omni.*
- Audio renames   Parakeet audio encoder requires significant tensor key surgery:
                    ffn1_* / ffn2_*  ->  ffn_* / ffn_*_1
                    attn_bias_u/v    ->  pos_bias_u/v
                    attn_rel_k       ->  attn_k_rel
                    attn_norm        ->  ln1
                    out_norm         ->  ln2 (weight only)
- Audio drops     Unmappable tensors stripped:
                    conv_bn.{bias,running_mean,running_var,weight}  (BN - no schema slot)
                    *.bias on all norms (schema only accepts .weight)
- No clamp synth  Parakeet does NOT use ClippableLinear - no input_min/max tensors needed

What this plugin does NOT do (handled by base / engine)
--------------------------------------------------------
- QKV splitting (Parakeet does not fuse QKV)
- Clamp scalar synthesis (unnecessary for Parakeet)
- LLM SSM tensor renames (nemotron_h uses standard gguf SSM names)

KV source map
-------------
llm_fields (nemotron_h.*)
    -> nemotron_h_omni.embedding_length
    -> nemotron_h_omni.block_count
    -> nemotron_h_omni.feed_forward_length         (per-layer int array)
    -> nemotron_h_omni.attention.head_count
    -> nemotron_h_omni.attention.head_count_kv
    -> nemotron_h_omni.attention.key_length        (head_dim)
    -> nemotron_h_omni.attention.value_length      (head_dim)
    -> nemotron_h_omni.ssm.conv_kernel
    -> nemotron_h_omni.ssm.inner_size
    -> nemotron_h_omni.ssm.state_size
    -> nemotron_h_omni.ssm.time_step_rank          (n_heads in NemotronH)
    -> nemotron_h_omni.ssm.group_count
    -> nemotron_h_omni.context_length
    -> nemotron_h_omni.rope.freq_base              (absent on pure-Mamba builds)
    -> nemotron_h_omni.rope.scaling.finetuned
    -> nemotron_h_omni.attention.layer_norm_rms_epsilon
    -> nemotron_h_omni.expert_* (MoE only)
    -> all tokenizer.ggml.* fields
mmproj_fields (clip.*)
    -> nemotron_h_omni.vision.*
    -> nemotron_h_omni.audio.*
hardcoded
    -> tokenizer.ggml.add_eos_token = False

TODO / TBD (requires blob read from ollama.com/library/nemotron3:33b)
----------------------------------------------------------------------
- Exact KV keys the Ollama nemotron_h_omni loader expects
- Whether rope.* keys are required (NemotronH sets use_rope=False for hybrid)
- audio.conv_kernel_size hparam name for 9-tap Parakeet kernel
- Vision projector_type re-injection as NEMOTRON_V2_VL
- Top-level audio projection tensor renames (a.pre_encode.out, etc.)
"""

from __future__ import annotations

import re
import sys

import numpy as np
from gguf import GGUFWriter, GGUFValueType, GGMLQuantizationType

from .base import (
    STATUS_EXPERIMENTAL,
    BaseModelCore,
    FLOAT_TYPES,
    _read_array,
    _read_scalar,
    copy_field,
    write_tensor,
)

# ---------------------------------------------------------------------------
# Audio tensor drops - no schema slot exists for these
# ---------------------------------------------------------------------------

_AUDIO_DROP_SUFFIXES: frozenset[str] = frozenset({
    # Batch-normalisation (Parakeet SSConvEncoder uses BN; MMPROJ schema has no slot)
    "conv_bn.bias",
    "conv_bn.running_mean",
    "conv_bn.running_var",
    "conv_bn.weight",
    # Norm .bias (schema only accepts .weight for all audio norms)
    "attn_norm.bias",
    "conv_norm.bias",
    "ffn1_norm.bias",
    "ffn2_norm.bias",
    "out_norm.bias",
})

# ---------------------------------------------------------------------------
# Per-block audio tensor renames  (suffix -> new suffix)
# Corrects Parakeet naming to llama.cpp MMPROJ schema names.
# ---------------------------------------------------------------------------

_AUDIO_BLK_RENAMES: dict[str, str] = {
    # Attention norms
    "attn_norm.weight":      "ln1.weight",          # A_ENC_INPUT_NORM   -> a.blk.N.ln1
    "out_norm.weight":       "ln2.weight",           # A_ENC_OUTPUT_NORM  -> a.blk.N.ln2

    # Positional bias tensors (no .weight suffix - bare tensor names)
    "attn_bias_u":           "pos_bias_u",           # A_ENC_POS_BIAS_U
    "attn_bias_v":           "pos_bias_v",           # A_ENC_POS_BIAS_V

    # Relative key projection
    "attn_rel_k.weight":     "attn_k_rel.weight",    # A_ENC_ATTN_K_REL

    # FFN first sub-block: ffn1_* -> ffn_*
    "ffn1_down.weight":      "ffn_down.weight",      # A_ENC_FFN_DOWN
    "ffn1_up.weight":        "ffn_up.weight",        # A_ENC_FFN_UP
    "ffn1_norm.weight":      "ffn_norm.weight",      # A_ENC_FFN_NORM

    # FFN second sub-block: ffn2_* -> ffn_*_1
    "ffn2_down.weight":      "ffn_down_1.weight",    # A_ENC_FFN_DOWN_1
    "ffn2_up.weight":        "ffn_up_1.weight",      # A_ENC_FFN_UP_1
    "ffn2_norm.weight":      "ffn_norm_1.weight",    # A_ENC_FFN_NORM_1
}

# ---------------------------------------------------------------------------
# Top-level (non-block) audio tensor renames
# TBD: fill in after reading actual mmproj.gguf tensor names
# ---------------------------------------------------------------------------

_AUDIO_TOP_RENAMES: dict[str, str] = {
    # Example (verify against actual GGUF before enabling):
    # "a.pre_encode.out.weight": "mm.a.fc.weight",
    # "a.input_projection.weight": "a.pre_encode.out.weight",
}

# ---------------------------------------------------------------------------
# Nemotron3OmniModelCore
# ---------------------------------------------------------------------------

class Nemotron3OmniModelCore(BaseModelCore):
    """Merge plugin for NVIDIA Nemotron 3 Nano Omni (30B-A3B, RADIO vision + Parakeet audio)."""

    MODEL_TYPE: str = "nemotron3omni"
    REQUIRES_BLOB: bool = False
    STATUS: str = STATUS_EXPERIMENTAL

    @classmethod
    def get_help_info(cls):
        return {
            "description": "Nemotron 3 Nano Omni 30B-A3B (vision RADIO + audio Parakeet)",
            "requires_blob": False,
            "status": STATUS_EXPERIMENTAL,
            "extra_options": [
                ("--vision", "Include RADIO vision encoder tensors and KV"),
                ("--audio",  "Include Parakeet audio encoder tensors and KV"),
            ],
        }

    def __init__(self, arch: str) -> None:
        super().__init__(arch)
        # Detected in prepare_llm() — "nemotron_h" or "nemotron_h_moe"
        self._llm_arch: str = "nemotron_h"
        self._is_moe: bool = False

    # ---- CLI ---------------------------------------------------------------

    @classmethod
    def add_args(cls, parser) -> None:
        g = parser.add_argument_group("Nemotron 3 Omni options")
        g.add_argument("--vision", action="store_true", default=False,
                       help="Include RADIO vision encoder tensors and KV.")
        g.add_argument("--audio",  action="store_true", default=False,
                       help="Include Parakeet audio encoder tensors and KV.")

    @classmethod
    def format_args_summary(cls, args) -> str | None:
        return (
            f"Nemotron 3 Omni Multimodal functions:\n"
            f"  Vision (RADIO):    {'Enabled' if args.vision else 'Disabled'}\n"
            f"  Audio (Parakeet):  {'Enabled' if args.audio else 'Disabled'}\n"
        )

    @classmethod
    def validate_args(cls, args) -> None:
        if not args.vision and not args.audio:
            sys.exit("ERROR: nemotron3omni requires at least one of --vision or --audio.")

    # ---- Pre-scan LLM ------------------------------------------------------

    def prepare_llm(self, llm) -> None:
        """Detect LLM arch variant so inject_kv reads the correct llm_fields keys."""
        arch_field = llm.fields.get("general.architecture")
        if arch_field is not None:
            detected = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
            self._llm_arch = detected
            self._is_moe = (detected == "nemotron_h_moe")
            print(f"  LLM arch detected: {detected} (MoE={self._is_moe})")

    # ---- KV drop -----------------------------------------------------------

    def get_kv_drop(self) -> set[str]:
        """
        Drop all nemotron_h.* keys from LLM passthrough.
        They are re-injected under nemotron_h_omni.* by inject_kv().

        Core difference from Gemma4: the LLM arch string differs from the
        output arch string, so ALL arch-prefixed LLM KV must be dropped
        and re-written under the new prefix — there is no passthrough alias.
        """
        a_out = self.arch  # "nemotron_h_omni"

        _NH_SUFFIXES = (
            "embedding_length", "block_count", "context_length", "feed_forward_length",
            "attention.head_count", "attention.head_count_kv",
            "attention.key_length", "attention.value_length",
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel", "ssm.inner_size", "ssm.state_size",
            "ssm.time_step_rank", "ssm.group_count",
            "rope.freq_base", "rope.scaling.finetuned",
            "expert_count", "expert_used_count", "expert_shared_count",
            "expert_feed_forward_length", "expert_shared_feed_forward_length",
            "expert_weights_norm", "expert_weights_scale",
            "expert_group_count", "moe_latent_size",
        )
        extra: set[str] = set()
        for s in _NH_SUFFIXES:
            extra.add(f"nemotron_h.{s}")
            extra.add(f"nemotron_h_moe.{s}")

        extra |= {
            # Tokenizer (re-injected explicitly)
            "tokenizer.ggml.eos_token_ids", "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.add_eos_token", "tokenizer.ggml.add_padding_token",
            "tokenizer.ggml.add_mask_token", "tokenizer.ggml.add_unknown_token",
            "tokenizer.ggml.model", "tokenizer.ggml.pre",
            "tokenizer.ggml.scores", "tokenizer.ggml.token_type",
            # Vision mmproj KV (sourced from mmproj_fields)
            f"{a_out}.vision.feed_forward_length",
            f"{a_out}.vision.num_channels",
            f"{a_out}.vision.projector.scale_factor",
            # Audio mmproj KV (sourced from mmproj_fields)
            f"{a_out}.audio.attention.head_count",
            f"{a_out}.audio.attention.layer_norm_epsilon",
            f"{a_out}.audio.block_count",
            f"{a_out}.audio.conv_kernel_size",
            f"{a_out}.audio.embedding_length",
            f"{a_out}.audio.feed_forward_length",
        }
        return super().get_kv_drop() | extra

    # ---- KV renames --------------------------------------------------------

    def get_kv_renames(self) -> dict[str, str]:
        """clip.vision.* -> nemotron_h_omni.vision.*"""
        a = self.arch
        return {
            "clip.vision.block_count":                  f"{a}.vision.block_count",
            "clip.vision.embedding_length":             f"{a}.vision.embedding_length",
            "clip.vision.attention.head_count":         f"{a}.vision.attention.head_count",
            "clip.vision.attention.layer_norm_epsilon": f"{a}.vision.attention.layer_norm_epsilon",
            "clip.vision.patch_size":                   f"{a}.vision.patch_size",
            "clip.vision.image_size":                   f"{a}.vision.image_size",
            "clip.vision.image_mean":                   f"{a}.vision.image_mean",
            "clip.vision.image_std":                    f"{a}.vision.image_std",
            # TBD RADIO-specific:
            # "clip.vision.spatial_merge_size":         f"{a}.vision.spatial_merge_size",
        }

    # ---- mmproj KV conditional filter --------------------------------------

    def should_skip_mmproj_kv(self, field_name: str, renamed_key: str, args) -> bool:
        a = self.arch
        if not args.vision and (
            field_name.startswith("clip.vision.")
            or renamed_key.startswith(f"{a}.vision.")
        ):
            return True
        # Audio KV re-injected explicitly in inject_kv — suppress passthrough
        if field_name.startswith("clip.audio."):
            return True
        return False

    # ---- KV injection ------------------------------------------------------

    def inject_kv(
        self,
        writer: GGUFWriter,
        ref_fields: dict | None,
        mmproj_fields: dict,
        llm_fields: dict,
        *,
        args,
    ) -> None:
        """
        Re-inject all arch-critical KV fields under nemotron_h_omni.*.

        The LLM GGUF uses arch "nemotron_h" (or "nemotron_h_moe"), so
        every read from llm_fields uses self._llm_arch as the prefix,
        while every write to writer uses self.arch ("nemotron_h_omni").

        This prefix remapping is the primary structural difference from
        the Gemma4 plugin, where LLM arch == output arch.
        """
        a_in  = self._llm_arch   # "nemotron_h" or "nemotron_h_moe"
        a_out = self.arch        # "nemotron_h_omni"

        def _llm(key: str):
            return _read_scalar(llm_fields, f"{a_in}.{key}")

        def _llm_arr(key: str):
            return _read_array(llm_fields, f"{a_in}.{key}")

        def _llm_has(key: str) -> bool:
            return f"{a_in}.{key}" in llm_fields

        # -- Core model dimensions -------------------------------------------
        writer.add_uint32(f"{a_out}.embedding_length", int(_llm("embedding_length")))
        writer.add_uint32(f"{a_out}.block_count",      int(_llm("block_count")))
        writer.add_uint32(f"{a_out}.context_length",   int(_llm("context_length")))

        ffl_key = f"{a_in}.feed_forward_length"
        if ffl_key in llm_fields:
            ffl_field = llm_fields[ffl_key]
            if ffl_field.types[0] == GGUFValueType.ARRAY:
                writer.add_array(f"{a_out}.feed_forward_length",
                                 [int(x) for x in _llm_arr("feed_forward_length")])
            else:
                writer.add_uint32(f"{a_out}.feed_forward_length",
                                  int(_llm("feed_forward_length")))

        # -- Attention -------------------------------------------------------
        hckv_key = f"{a_in}.attention.head_count_kv"
        if hckv_key in llm_fields:
            hckv_field = llm_fields[hckv_key]
            if hckv_field.types[0] == GGUFValueType.ARRAY:
                writer.add_array(f"{a_out}.attention.head_count_kv",
                                 _llm_arr("attention.head_count_kv"))
            else:
                writer.add_uint32(f"{a_out}.attention.head_count_kv",
                                  int(_llm("attention.head_count_kv")))

        writer.add_uint32(f"{a_out}.attention.head_count",
                          int(_llm("attention.head_count")))
        writer.add_uint32(f"{a_out}.attention.key_length",
                          int(_llm("attention.key_length")))
        writer.add_uint32(f"{a_out}.attention.value_length",
                          int(_llm("attention.value_length")))
        writer.add_float32(f"{a_out}.attention.layer_norm_rms_epsilon",
                           float(_llm("attention.layer_norm_rms_epsilon")))

        # -- SSM (Mamba2) ----------------------------------------------------
        writer.add_uint32(f"{a_out}.ssm.conv_kernel",    int(_llm("ssm.conv_kernel")))
        writer.add_uint32(f"{a_out}.ssm.inner_size",     int(_llm("ssm.inner_size")))
        writer.add_uint32(f"{a_out}.ssm.state_size",     int(_llm("ssm.state_size")))
        writer.add_uint32(f"{a_out}.ssm.time_step_rank", int(_llm("ssm.time_step_rank")))
        writer.add_uint32(f"{a_out}.ssm.group_count",    int(_llm("ssm.group_count")))

        # -- RoPE (optional) -------------------------------------------------
        # NemotronH sets rope.scaling.finetuned=False and may omit rope.freq_base
        # for hybrid builds where all SSM blocks skip RoPE entirely.
        if _llm_has("rope.freq_base"):
            writer.add_float32(f"{a_out}.rope.freq_base",
                               float(_llm("rope.freq_base")))
        if _llm_has("rope.scaling.finetuned"):
            writer.add_bool(f"{a_out}.rope.scaling.finetuned",
                            bool(_llm("rope.scaling.finetuned")))

        # -- MoE extras (nemotron_h_moe only) --------------------------------
        if self._is_moe:
            for uk in ("expert_count", "expert_used_count", "expert_shared_count",
                       "expert_feed_forward_length", "expert_shared_feed_forward_length",
                       "expert_group_count", "moe_latent_size"):
                if _llm_has(uk):
                    writer.add_uint32(f"{a_out}.{uk}", int(_llm(uk)))
            for fk in ("expert_weights_norm", "expert_weights_scale"):
                if _llm_has(fk):
                    writer.add_float32(f"{a_out}.{fk}", float(_llm(fk)))

        # -- Vision KV -------------------------------------------------------
        if args.vision:
            _scale_key = "clip.vision.pooling_kernel_size"
            scale_factor = (
                int(_read_scalar(mmproj_fields, _scale_key))
                if _scale_key in mmproj_fields
                else 2  # RADIO: downsample_ratio=0.5 -> scale_factor=2
            )
            writer.add_uint32(f"{a_out}.vision.projector.scale_factor", scale_factor)

            _vffl = "clip.vision.feed_forward_length"
            if _vffl in mmproj_fields:
                writer.add_uint32(f"{a_out}.vision.feed_forward_length",
                                  int(_read_scalar(mmproj_fields, _vffl)))

            _vch = "clip.vision.num_channels"
            writer.add_uint32(f"{a_out}.vision.num_channels",
                              int(_read_scalar(mmproj_fields, _vch))
                              if _vch in mmproj_fields else 3)

            # TBD: re-inject clip.projector_type as "nemotron_v2_vl"?
            # writer.add_string(f"{a_out}.vision.projector_type", "nemotron_v2_vl")

        # -- Audio KV --------------------------------------------------------
        if args.audio:
            _akv: list[tuple[str, str, str]] = [
                ("clip.audio.attention.head_count",
                 f"{a_out}.audio.attention.head_count",         "uint32"),
                ("clip.audio.attention.layer_norm_epsilon",
                 f"{a_out}.audio.attention.layer_norm_epsilon", "float32"),
                ("clip.audio.block_count",
                 f"{a_out}.audio.block_count",                  "uint32"),
                ("clip.audio.conv_kernel_size",
                 f"{a_out}.audio.conv_kernel_size",             "uint32"),
                ("clip.audio.embedding_length",
                 f"{a_out}.audio.embedding_length",             "uint32"),
                ("clip.audio.feed_forward_length",
                 f"{a_out}.audio.feed_forward_length",          "uint32"),
                # TBD: ("clip.audio.num_mel_bins",
                #        f"{a_out}.audio.num_mel_bins", "uint32"),
            ]
            for src, out, vtype in _akv:
                if src not in mmproj_fields:
                    if src == "clip.audio.attention.layer_norm_epsilon":
                        writer.add_float32(out, 1e-5)  # Parakeet default
                        print(f"  NOTE: {src} absent - using Parakeet default 1e-5")
                    else:
                        print(f"  WARNING: audio KV '{src}' not in mmproj - skipping")
                    continue
                val = _read_scalar(mmproj_fields, src)
                writer.add_uint32(out, int(val)) if vtype == "uint32" \
                    else writer.add_float32(out, float(val))

        # -- Tokenizer -------------------------------------------------------
        writer.add_bool("tokenizer.ggml.add_eos_token", False)

        if "tokenizer.ggml.eos_token_ids" in llm_fields:
            writer.add_array("tokenizer.ggml.eos_token_ids",
                             [int(x) for x in _read_array(llm_fields,
                              "tokenizer.ggml.eos_token_ids")])
        elif "tokenizer.ggml.eos_token_id" in llm_fields:
            eid = int(_read_scalar(llm_fields, "tokenizer.ggml.eos_token_id"))
            writer.add_array("tokenizer.ggml.eos_token_ids", [eid])

        if "tokenizer.ggml.eos_token_id" in llm_fields:
            writer.add_uint32("tokenizer.ggml.eos_token_id",
                              int(_read_scalar(llm_fields, "tokenizer.ggml.eos_token_id")))

        for sk in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
            if sk in llm_fields:
                f = llm_fields[sk]
                writer.add_string(sk, bytes(f.parts[f.data[0]]).decode("utf-8"))

        for ak in ("tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
            if ak in llm_fields:
                copy_field(writer, llm_fields[ak], name=ak)

        for bk in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_padding_token",
                   "tokenizer.ggml.add_mask_token", "tokenizer.ggml.add_unknown_token"):
            if bk in llm_fields:
                writer.add_bool(bk, bool(_read_scalar(llm_fields, bk)))

        if "tokenizer.ggml.bos_token_id" in llm_fields:
            writer.add_uint32("tokenizer.ggml.bos_token_id",
                              int(_read_scalar(llm_fields, "tokenizer.ggml.bos_token_id")))

        _ft = (int(_read_scalar(llm_fields, "general.file_type"))
               if "general.file_type" in llm_fields else 32)
        writer.add_uint32("general.file_type", _ft)

    # ---- mmproj tensor processing ------------------------------------------

    def process_mmproj_tensors(self, mmproj, args) -> dict:
        """
        Load Nemotron 3 Omni mmproj tensors:
        - Modality filtering (--vision / --audio)
        - Audio tensor drops  (BN stats + norm biases — no schema slot)
        - Audio tensor renames (Parakeet names -> llama.cpp MMPROJ schema)
        - Vision passthrough  (RADIO ViT-H tensors — no rename needed)

        No clamp scalar injection needed. Unlike Gemma4's ClippableLinear,
        Parakeet has no .input_min/.input_max activation clamp tensors.
        """
        mmproj_names = {t.name for t in mmproj.tensors}
        has_audio  = any(n.startswith("a.") or n.startswith("mm.a.") for n in mmproj_names)
        has_vision = any(n.startswith("v.") for n in mmproj_names)

        if args.audio and not has_audio:
            sys.exit(
                "ERROR: --audio specified but mmproj has no audio tensors (a.*). "
                "Ensure you are using the Omni mmproj, not the VL-only mmproj."
            )
        if args.vision and not has_vision:
            sys.exit("ERROR: --vision specified but mmproj has no vision tensors (v.*).")
        if has_audio and not args.audio:
            print("  NOTE: mmproj has audio tensors but --audio not set; audio will be stripped.")
        if has_vision and not args.vision:
            print("  NOTE: mmproj has vision tensors but --vision not set; vision will be stripped.")

        encoder_tensors: dict = {}
        skipped_audio = skipped_vision = renamed_count = dropped_count = 0

        for t in mmproj.tensors:
            is_audio  = t.name.startswith("a.") or t.name.startswith("mm.a.")
            is_vision = t.name.startswith("v.") or (
                t.name.startswith("mm.") and not is_audio
            )

            if is_audio:
                if args.audio:
                    final_name, dropped = _nemotron_omni_audio_rename(t.name)
                    if dropped:
                        dropped_count += 1
                        print(f"  tensor drop:   {t.name}  (no schema slot)")
                        continue
                    if final_name != t.name:
                        print(f"  tensor rename: {t.name} -> {final_name}")
                        renamed_count += 1
                    encoder_tensors[final_name] = t
                else:
                    skipped_audio += 1
            elif is_vision:
                if args.vision:
                    encoder_tensors[t.name] = t
                else:
                    skipped_vision += 1
            else:
                encoder_tensors[t.name] = t  # unknown prefix - include to be safe

        print(f"  Encoder tensors included : {len(encoder_tensors)}")
        if renamed_count:
            print(f"  Audio tensors renamed    : {renamed_count}")
        if dropped_count:
            print(f"  Audio tensors dropped    : {dropped_count}")
        if skipped_audio:
            print(f"  Audio tensors stripped   : {skipped_audio}")
        if skipped_vision:
            print(f"  Vision tensors stripped  : {skipped_vision}")

        return encoder_tensors

    # post_write_tensors: no override needed.
    # Parakeet has no ClippableLinear layers - no clamp scalars to synthesise.


# ---------------------------------------------------------------------------
# Module-level audio tensor rename helper
# ---------------------------------------------------------------------------

def _nemotron_omni_audio_rename(name: str) -> tuple[str, bool]:
    """
    Rename a single Parakeet audio tensor from its llama.cpp GGUF name
    to the Ollama-expected name, signalling whether to drop it entirely.

    Returns: (final_name, should_drop)

    Key differences vs _gemma4_audio_rename():
    - Much larger rename set (ffn1_/ffn2_ duality, attn_norm, out_norm)
    - Active DROP logic for BN tensors and norm .bias tensors
    - No ln2 -> layer_pre_norm hop (that was a Gemma4-specific Ollama quirk)
    """
    m = re.match(r"(a\.blk\.\d+\.)(.*)", name)
    if m:
        prefix, suffix = m.group(1), m.group(2)
        if suffix in _AUDIO_DROP_SUFFIXES:
            return name, True  # signal drop
        if suffix in _AUDIO_BLK_RENAMES:
            return prefix + _AUDIO_BLK_RENAMES[suffix], False
        return name, False

    if name in _AUDIO_TOP_RENAMES:
        return _AUDIO_TOP_RENAMES[name], False

    return name, False
