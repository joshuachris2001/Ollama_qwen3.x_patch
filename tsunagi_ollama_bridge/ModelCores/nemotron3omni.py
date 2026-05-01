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
- Audio renames   Parakeet audio encoder tensor key surgery via
                  _nemotron_omni_audio_rename():
                    ffn1_* / ffn2_*  ->  ffn_* / ffn_*_1
                    attn_bias_u/v    ->  pos_bias_u/v
                    attn_rel_k       ->  attn_k_rel
                    attn_norm        ->  ln1
                    out_norm         ->  ln2 (weight only; bias dropped)
- Audio drops     Unmappable tensors stripped (_AUDIO_DROP_SUFFIXES):
                    conv_bn.{bias,running_mean,running_var,weight,num_batches_tracked}
                    *.bias on all norms (schema only accepts .weight)
- Audio reshapes  conv_dw: squeeze axis=0 ([1,K,C] -> [K,C])
                  conv_pw1/conv_pw2: squeeze axis=-1 ([C_out,C_in,1] -> [C_out,C_in])
- No clamp synth  Parakeet does NOT use ClippableLinear — no input_min/max needed

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
    -> nemotron_h_omni.attention.layer_norm_rms_epsilon
    -> nemotron_h_omni.attention.layer_norm_epsilon  (alias of rms_epsilon)
    -> nemotron_h_omni.ssm.conv_kernel
    -> nemotron_h_omni.ssm.inner_size
    -> nemotron_h_omni.ssm.state_size
    -> nemotron_h_omni.ssm.time_step_rank
    -> nemotron_h_omni.ssm.group_count
    -> nemotron_h_omni.context_length              (capped to 131072)
    -> nemotron_h_omni.rope.freq_base              (absent on pure-Mamba builds)
    -> nemotron_h_omni.rope.dimension_count        (hardcoded 128)
    -> nemotron_h_omni.rope.scaling.finetuned
    -> nemotron_h_omni.expert_* (MoE only)
    -> all tokenizer.ggml.* fields
mmproj_fields (clip.*)
    -> nemotron_h_omni.vision.*
    -> nemotron_h_omni.audio.*
hardcoded (Ollama blob-verified)
    -> tokenizer.ggml.add_eos_token = False
    -> tokenizer.ggml.pre = "default"
    -> tokenizer.ggml.eos_token_ids = [2, 11]
    -> general.parameter_count = 33013666128
    -> nemotron_h_omni.rope.dimension_count = 128
    -> nemotron_h_omni.context_length = 131072
    -> nemotron_h_omni.vision.image_size = 512
    -> nemotron_h_omni.vision.max_tiles = 12
    -> nemotron_h_omni.vision.min_num_patches = 1024
    -> nemotron_h_omni.vision.max_num_patches = 13312
    -> nemotron_h_omni.vision.use_thumbnail = True
    -> nemotron_h_omni.vision.image_token_id = 18
    -> nemotron_h_omni.vision.image_start_token_id = 19
    -> nemotron_h_omni.vision.image_end_token_id = 20
    -> nemotron_h_omni.audio.conv_kernel_size = 9
    -> nemotron_h_omni.audio.num_mel_bins = 128
    -> nemotron_h_omni.audio.sample_rate = 16000
    -> nemotron_h_omni.audio.subsampling_factor = 8
    -> nemotron_h_omni.audio.subsampling_conv_channels = 256
    -> nemotron_h_omni.audio.subsampling_conv_kernel_size = 3
    -> nemotron_h_omni.audio.subsampling_conv_stride = 2
    -> nemotron_h_omni.audio.projection_hidden_size = 4096
    -> nemotron_h_omni.audio.scale_input = False
    -> nemotron_h_omni.audio.sound_token_id = 27
"""

from __future__ import annotations

import copy
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
# Audio tensor drop suffixes — no GGUF schema slot exists for these
# ---------------------------------------------------------------------------

_AUDIO_DROP_SUFFIXES: tuple[str, ...] = (
    # Batch-norm stats — Parakeet exports these but GGUF has no slot
    "conv_bn.weight",
    "conv_bn.bias",
    "conv_bn.running_mean",
    "conv_bn.running_var",
    "conv_bn.num_batches_tracked",
    # Norm bias tensors — schema only accepts .weight for all audio norms
    "attn_norm.bias",
    "out_norm.bias",
    "ffn1_norm.bias",
    "ffn2_norm.bias",
    "conv_norm.bias",
)

# ---------------------------------------------------------------------------
# Nemotron3OmniModelCore
# ---------------------------------------------------------------------------

class Nemotron3OmniModelCore(BaseModelCore):
    """Merge plugin for NVIDIA Nemotron 3 Nano Omni (30B-A3B, RADIO vision + Parakeet audio)."""

    MODEL_TYPE: str = "nemotron_h_omni"
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
        self._llm_arch: str = "nemotron_h_moe"
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
            f"  Audio (Parakeet):  {'Enabled' if args.audio  else 'Disabled'}\n"
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
        Drop all nemotron_h.* / nemotron_h_moe.* keys from LLM passthrough
        and all clip.audio.* / re-injected nemotron_h_omni.* keys from mmproj.

        They are all re-written under nemotron_h_omni.* by inject_kv().
        """
        a_out = self.arch  # "nemotron_h_omni"

        _NH_SUFFIXES = (
            "embedding_length", "block_count", "context_length", "feed_forward_length",
            "attention.head_count", "attention.head_count_kv",
            "attention.key_length", "attention.value_length",
            "attention.layer_norm_rms_epsilon", "attention.layer_norm_epsilon",
            "ssm.conv_kernel", "ssm.inner_size", "ssm.state_size",
            "ssm.time_step_rank", "ssm.group_count",
            "rope.freq_base", "rope.scaling.finetuned", "rope.dimension_count",
            "expert_count", "expert_used_count", "expert_shared_count",
            "expert_feed_forward_length", "expert_shared_feed_forward_length",
            "expert_weights_norm", "expert_weights_scale",
            "expert_group_count", "expert_group_used_count",
            "moe_latent_size",
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
            # general (re-injected as hardcoded)
            "general.parameter_count",
            # Vision mmproj KV (sourced from mmproj_fields or hardcoded)
            f"{a_out}.vision.feed_forward_length",
            f"{a_out}.vision.num_channels",
            f"{a_out}.vision.projector.scale_factor",
            f"{a_out}.vision.image_size",
            f"{a_out}.vision.max_tiles",
            f"{a_out}.vision.min_num_patches",
            f"{a_out}.vision.max_num_patches",
            f"{a_out}.vision.use_thumbnail",
            f"{a_out}.vision.image_token_id",
            f"{a_out}.vision.image_start_token_id",
            f"{a_out}.vision.image_end_token_id",
            # Audio mmproj KV (sourced from mmproj_fields or hardcoded)
            f"{a_out}.audio.attention.head_count",
            f"{a_out}.audio.attention.layer_norm_epsilon",
            f"{a_out}.audio.block_count",
            f"{a_out}.audio.conv_kernel_size",
            f"{a_out}.audio.embedding_length",
            f"{a_out}.audio.feed_forward_length",
            f"{a_out}.audio.num_mel_bins",
            f"{a_out}.audio.sample_rate",
            f"{a_out}.audio.subsampling_factor",
            f"{a_out}.audio.subsampling_conv_channels",
            f"{a_out}.audio.subsampling_conv_kernel_size",
            f"{a_out}.audio.subsampling_conv_stride",
            f"{a_out}.audio.projection_hidden_size",
            f"{a_out}.audio.scale_input",
            f"{a_out}.audio.sound_token_id",
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
        # Cap context_length to 131072 (Ollama blob value; raw LLM may say 1048576)
        writer.add_uint32(f"{a_out}.context_length", 131072)

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
        writer.add_uint32(f"{a_out}.attention.head_count",
                          int(_llm("attention.head_count")))

        hckv_key = f"{a_in}.attention.head_count_kv"
        if hckv_key in llm_fields:
            hckv_field = llm_fields[hckv_key]
            if hckv_field.types[0] == GGUFValueType.ARRAY:
                writer.add_array(f"{a_out}.attention.head_count_kv",
                                 _llm_arr("attention.head_count_kv"))
            else:
                writer.add_uint32(f"{a_out}.attention.head_count_kv",
                                  int(_llm("attention.head_count_kv")))

        writer.add_uint32(f"{a_out}.attention.key_length",
                          int(_llm("attention.key_length")))
        writer.add_uint32(f"{a_out}.attention.value_length",
                          int(_llm("attention.value_length")))

        rms_eps = float(_llm("attention.layer_norm_rms_epsilon"))
        writer.add_float32(f"{a_out}.attention.layer_norm_rms_epsilon", rms_eps)
        # Ollama also writes the non-rms alias with the same value
        writer.add_float32(f"{a_out}.attention.layer_norm_epsilon", rms_eps)

        # -- RoPE ------------------------------------------------------------
        # NemotronH hybrid: some builds omit rope.freq_base entirely
        writer.add_uint32(f"{a_out}.rope.dimension_count", 128)
        if _llm_has("rope.freq_base"):
            writer.add_float32(f"{a_out}.rope.freq_base",
                               float(_llm("rope.freq_base")))
        if _llm_has("rope.scaling.finetuned"):
            writer.add_bool(f"{a_out}.rope.scaling.finetuned",
                            bool(_llm("rope.scaling.finetuned")))

        # -- SSM (Mamba2) ----------------------------------------------------
        writer.add_uint32(f"{a_out}.ssm.conv_kernel",    int(_llm("ssm.conv_kernel")))
        writer.add_uint32(f"{a_out}.ssm.inner_size",     int(_llm("ssm.inner_size")))
        writer.add_uint32(f"{a_out}.ssm.state_size",     int(_llm("ssm.state_size")))
        writer.add_uint32(f"{a_out}.ssm.time_step_rank", int(_llm("ssm.time_step_rank")))
        writer.add_uint32(f"{a_out}.ssm.group_count",    int(_llm("ssm.group_count")))

        # -- MoE extras (nemotron_h_moe only) --------------------------------
        if self._is_moe:
            for uk in (
                "expert_count", "expert_used_count", "expert_shared_count",
                "expert_feed_forward_length", "expert_shared_feed_forward_length",
                "expert_group_count", "expert_group_used_count", "moe_latent_size",
            ):
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

            # Hardcoded Ollama blob constants for RADIO vision
            writer.add_uint32(f"{a_out}.vision.image_size", 512)
            writer.add_uint32(f"{a_out}.vision.max_tiles", 12)
            writer.add_uint32(f"{a_out}.vision.min_num_patches", 1024)
            writer.add_uint32(f"{a_out}.vision.max_num_patches", 13312)
            writer.add_bool(f"{a_out}.vision.use_thumbnail", True)
            writer.add_uint32(f"{a_out}.vision.image_token_id", 18)
            writer.add_uint32(f"{a_out}.vision.image_start_token_id", 19)
            writer.add_uint32(f"{a_out}.vision.image_end_token_id", 20)

        # -- Audio KV --------------------------------------------------------
        if args.audio:
            _akv: list[tuple[str, str, str]] = [
                ("clip.audio.attention.head_count",
                 f"{a_out}.audio.attention.head_count",         "uint32"),
                ("clip.audio.attention.layer_norm_epsilon",
                 f"{a_out}.audio.attention.layer_norm_epsilon", "float32"),
                ("clip.audio.block_count",
                 f"{a_out}.audio.block_count",                  "uint32"),
                ("clip.audio.embedding_length",
                 f"{a_out}.audio.embedding_length",             "uint32"),
                ("clip.audio.feed_forward_length",
                 f"{a_out}.audio.feed_forward_length",          "uint32"),
            ]
            for src, out, vtype in _akv:
                if src not in mmproj_fields:
                    if src == "clip.audio.attention.layer_norm_epsilon":
                        writer.add_float32(out, 1e-5)  # Parakeet default
                        print(f"  NOTE: {src} absent — using Parakeet default 1e-5")
                    else:
                        print(f"  WARNING: audio KV '{src}' not in mmproj — skipping")
                    continue
                val = _read_scalar(mmproj_fields, src)
                writer.add_uint32(out, int(val)) if vtype == "uint32" \
                    else writer.add_float32(out, float(val))

            # Hardcoded Ollama blob constants for Parakeet audio
            writer.add_uint32(f"{a_out}.audio.conv_kernel_size", 9)
            writer.add_uint32(f"{a_out}.audio.num_mel_bins", 128)
            writer.add_uint32(f"{a_out}.audio.sample_rate", 16000)
            writer.add_uint32(f"{a_out}.audio.subsampling_factor", 8)
            writer.add_uint32(f"{a_out}.audio.subsampling_conv_channels", 256)
            writer.add_uint32(f"{a_out}.audio.subsampling_conv_kernel_size", 3)
            writer.add_uint32(f"{a_out}.audio.subsampling_conv_stride", 2)
            writer.add_uint32(f"{a_out}.audio.projection_hidden_size", 4096)
            writer.add_bool(f"{a_out}.audio.scale_input", False)
            writer.add_uint32(f"{a_out}.audio.sound_token_id", 27)

        # -- Tokenizer -------------------------------------------------------
        writer.add_bool("tokenizer.ggml.add_eos_token", False)
        # Hardcoded Ollama blob values — override whatever the LLM GGUF carries
        writer.add_string("tokenizer.ggml.pre", "default")
        writer.add_array("tokenizer.ggml.eos_token_ids", [2, 11])

        if "tokenizer.ggml.eos_token_id" in llm_fields:
            writer.add_uint32("tokenizer.ggml.eos_token_id",
                              int(_read_scalar(llm_fields, "tokenizer.ggml.eos_token_id")))

        for sk in ("tokenizer.ggml.model",):
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

        # -- General metadata ------------------------------------------------
        _ft = (int(_read_scalar(llm_fields, "general.file_type"))
               if "general.file_type" in llm_fields else 32)
        writer.add_uint32("general.file_type", _ft)
        writer.add_uint64("general.parameter_count", 33_013_666_128)

    # ---- mmproj tensor processing ------------------------------------------

    def process_mmproj_tensors(self, mmproj, args) -> dict:  # pyright: ignore[reportMissingTypeArgument]
        """
        Load Nemotron 3 Omni mmproj tensors:
        - Modality filtering (--vision / --audio)
        - Audio tensor drops  (BN stats + norm biases — no schema slot)
        - Audio tensor renames (Parakeet names -> llama.cpp MMPROJ schema)
        - Audio tensor reshapes (conv_dw, conv_pw1, conv_pw2)
        - Vision tensor renames (mm.model.mlp.* -> mm.norm/mm.1/mm.2)
        - Vision tensor reshapes (v.position_embd squeeze, v.patch_embd flatten)
        - Vision QKV split (fused attn_qkv -> attn_q + attn_k + attn_v)
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

            # ----------------------------------------------------------------
            # Audio path
            # ----------------------------------------------------------------
            if is_audio:
                if not args.audio:
                    skipped_audio += 1
                    continue

                name = t.name

                # 1. Hard-drop tensors with no GGUF schema slot
                suffix = name.split(".", 2)[-1] if name.startswith("a.blk.") else ""
                if any(name.endswith(d) or suffix == d for d in _AUDIO_DROP_SUFFIXES):
                    dropped_count += 1
                    print(f"  tensor drop:   {name}  (no schema slot)")
                    continue

                # 2. Reshape before rename (operates on raw data in-place view)
                if name.startswith("a.blk."):
                    if ".conv_dw." in name and name.endswith(".weight"):
                        # Raw Parakeet: [1, K, C]  ->  [K, C]
                        if t.data.ndim == 3 and t.data.shape[0] == 1:
                            t.data = t.data.squeeze(axis=0)
                    elif (".conv_pw1." in name or ".conv_pw2." in name) and name.endswith(".weight"):
                        # Raw Parakeet: [C_out, C_in, 1]  ->  [C_out, C_in]
                        if t.data.ndim == 3 and t.data.shape[2] == 1:
                            t.data = t.data.squeeze(axis=-1)

                # 3. Rename
                final_name = _nemotron_omni_audio_rename(name)
                if final_name != name:
                    print(f"  tensor rename: {name} -> {final_name}")
                    renamed_count += 1

                encoder_tensors[final_name] = t

            # ----------------------------------------------------------------
            # Vision path
            # ----------------------------------------------------------------
            elif is_vision:
                if not args.vision:
                    skipped_vision += 1
                    continue

                name = t.name

                # Vision projector renames
                if name == "mm.model.mlp.0.weight":   name = "mm.norm.weight"
                elif name == "mm.model.mlp.0.bias":   name = "mm.norm.bias"
                elif name == "mm.model.mlp.1.weight": name = "mm.1.weight"
                elif name == "mm.model.mlp.1.bias":   name = "mm.1.bias"
                elif name == "mm.model.mlp.3.weight": name = "mm.2.weight"
                elif name == "mm.model.mlp.3.bias":   name = "mm.2.bias"
                elif name == "v.class_embd":           name = "v.cls_embd"

                # Position embed: rename + squeeze [1, seq, dim] -> [seq, dim]
                elif name == "v.position_embd.weight":
                    name = "v.position_embd"
                    if t.data.ndim == 3 and t.data.shape[0] == 1:
                        t.data = t.data.squeeze(axis=0)

                # Patch embed: flatten spatial dims [H, W, C, D] -> [H*W*C, D]
                elif name == "v.patch_embd.weight":
                    t.data = t.data.reshape(-1, t.data.shape[-1])

                # QKV split: fused [3*D, D] -> three tensors [D, D]
                elif ".attn_qkv." in name:
                    chunks = np.split(t.data, 3, axis=0)
                    for part, suffix in zip(chunks, ("attn_q", "attn_k", "attn_v")):
                        clone = copy.copy(t)
                        clone.data = part
                        clone.name = name.replace("attn_qkv", suffix)
                        encoder_tensors[clone.name] = clone
                    continue  # skip adding the original fused tensor

                if name != t.name:
                    print(f"  tensor rename: {t.name} -> {name}")
                    renamed_count += 1

                encoder_tensors[name] = t

            # ----------------------------------------------------------------
            # Unknown prefix — include to be safe
            # ----------------------------------------------------------------
            else:
                encoder_tensors[t.name] = t

        print(f"  Encoder tensors included : {len(encoder_tensors)}")
        if renamed_count:
            print(f"  Tensors renamed          : {renamed_count}")
        if dropped_count:
            print(f"  Audio tensors dropped    : {dropped_count}")
        if skipped_audio:
            print(f"  Audio tensors stripped   : {skipped_audio}")
        if skipped_vision:
            print(f"  Vision tensors stripped  : {skipped_vision}")

        return encoder_tensors

    # post_write_tensors: no override needed.
    # Parakeet has no ClippableLinear layers — no clamp scalars to synthesise.


# ---------------------------------------------------------------------------
# Audio tensor rename helper
# ---------------------------------------------------------------------------

def _nemotron_omni_audio_rename(name: str) -> str:
    """
    Rename a single mmproj Parakeet audio tensor from its raw GGUF name
    (as exported by the Nemotron 3 Omni mmproj.gguf) to the expected
    Ollama / llama.cpp schema name.

    Per-block renames (a.blk.N.xxx -> a.blk.N.yyy):
      Raw mmproj name          Ollama schema name       GGUF MODEL_TENSOR
      attn_norm.weight    ->   ln1.weight               A_ENC_INPUT_NORM
      out_norm.weight     ->   ln2.weight               A_ENC_OUTPUT_NORM
      ffn1_norm.weight    ->   ffn_norm.weight          A_ENC_FFN_NORM
      ffn2_norm.weight    ->   ffn_norm_1.weight        A_ENC_FFN_NORM_1
      ffn1_down.weight    ->   ffn_down.weight          A_ENC_FFN_DOWN
      ffn1_up.weight      ->   ffn_up.weight            A_ENC_FFN_UP
      ffn2_down.weight    ->   ffn_down_1.weight        A_ENC_FFN_DOWN_1
      ffn2_up.weight      ->   ffn_up_1.weight          A_ENC_FFN_UP_1
      attn_rel_k.weight   ->   attn_k_rel.weight        A_ENC_ATTN_K_REL
      attn_bias_u         ->   pos_bias_u               A_ENC_POS_BIAS_U
      attn_bias_v         ->   pos_bias_v               A_ENC_POS_BIAS_V

    Top-level projector renames:
      a.pre_encode.out.weight  ->  mm.a.fc.weight
      a.pre_encode.out.bias    ->  mm.a.fc.bias
      a.input_projection.weight -> a.pre_encode.out.weight
    """
    # Per-block renames
    m = re.match(r"(a\.blk\.\d+\.)(.*)", name)
    if m:
        prefix, suffix = m.group(1), m.group(2)
        blk_renames: dict[str, str] = {
            # Norm renames
            "attn_norm.weight":  "ln1.weight",
            "out_norm.weight":   "ln2.weight",
            "ffn1_norm.weight":  "ffn_norm.weight",
            "ffn2_norm.weight":  "ffn_norm_1.weight",
            # FFN linear renames
            "ffn1_down.weight":  "ffn_down.weight",
            "ffn1_up.weight":    "ffn_up.weight",
            "ffn2_down.weight":  "ffn_down_1.weight",
            "ffn2_up.weight":    "ffn_up_1.weight",
            # Relative-attention renames
            "attn_rel_k.weight": "attn_k_rel.weight",
            # Positional bias (no .weight suffix in schema)
            "attn_bias_u":       "pos_bias_u",
            "attn_bias_v":       "pos_bias_v",
        }
        if suffix in blk_renames:
            return prefix + blk_renames[suffix]

    # Top-level projector renames
    proj_renames: dict[str, str] = {
        "a.pre_encode.out.weight":   "mm.a.fc.weight",
        "a.pre_encode.out.bias":     "mm.a.fc.bias",
        "a.input_projection.weight": "a.pre_encode.out.weight",
    }
    if name in proj_renames:
        return proj_renames[name]

    return name
