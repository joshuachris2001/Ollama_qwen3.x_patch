"""
ModelCores/gemma4.py
====================
ModelCore plugin for Gemma 4 (all sizes: E2B, E4B, 26B MoE, 31B dense).

Gemma 4 specifics handled here
-------------------------------
• CLI flags      --vision / --audio to select which encoder(s) to include
• KV drop        Minimal — only keys that are re-injected or genuinely absent
• KV renames     Gemma4-specific subset (no spatial_merge_size, uses image_size)
• inject_kv      Sources all values from llm_fields, mmproj_fields, or hardcoded
                 constants.  No Ollama blob required.
• mmproj tensors Passthrough + audio tensor renames (ggml-org → Ollama naming)
• Clamp scalars  Synthesised from calibrated block-0 constants (E2B/E4B, identical)
                 plus ±FLT_MAX identity clamps for all other blocks.

What this plugin does NOT do (handled by base / engine)
--------------------------------------------------------
• QKV splitting        (Gemma4 mmproj does not fuse QKV)
• Deepstack handling   (Gemma4 has no deepstack vision layers)
• Patch-embed stacking (Gemma4 uses a standard single patch embed)
• LLM tensor renames   (Gemma4 needs none)

KV source map
-------------
llm_fields  → attention.head_count_kv, attention.key/value_length[_swa],
              attention.sliding_window[_pattern], attention.shared_kv_layers,
              embedding_length_per_layer_input, rope.dimension_count[_swa],
              rope.freq_base[_swa], final_logit_softcapping,
              feed_forward_length,
              all tokenizer.ggml.* fields
mmproj_fields → vision.feed_forward_length, vision.num_channels (default 3),
                audio.block_count, audio.embedding_length,
                audio.attention.head_count, audio.attention.layer_norm_epsilon,
                audio.conv_kernel_size, audio.feed_forward_length
hardcoded   → vision.projector.scale_factor = 3 (Go: if nMerge==0 { nMerge=3 })
              tokenizer.ggml.add_eos_token = False (Go always forces this)
              audio clamp block-0 constants (calibrated from Ollama E2B+E4B blobs)
              audio clamp all other blocks = ±FLT_MAX identity
"""

from __future__ import annotations

import re
import sys
#from typing import override

import numpy as np
from gguf import GGUFWriter, GGUFValueType, GGMLQuantizationType

from .base import (
    BaseModelCore,
    FLOAT_TYPES,
    _read_array,
    _read_scalar,
    copy_field,
    write_tensor,
)

# ---------------------------------------------------------------------------
# Activation-clamp scalar tensor suffixes (Gemma4ClippableLinear)
# ---------------------------------------------------------------------------

_CLAMP_SUFFIXES = (".input_min", ".input_max", ".output_min", ".output_max")

# ---------------------------------------------------------------------------
# Audio conformer block-0 clamp constants
#
# Source: direct read from Ollama blobs:
#   sha256-4e30e26... (gemma4:e2b)  sha256-4c27e0f... (gemma4:e4b)
# All 40 values are bit-for-bit identical between E2B and E4B, confirming
# these are shared architecture constants, not per-size learned parameters.
# Asymmetric ±X values are expected BF16 rounding artefacts.
#
# key → (input_min, input_max, output_min, output_max)
# ---------------------------------------------------------------------------

_AUDIO_CLAMP_BLK0: dict[str, tuple[float, float, float, float]] = {
    "attn_k":     (-20.375,   +20.250,   -34.500,  +34.250),
    "attn_out":   (-26.625,   +26.500,  -101.500, +100.500),
    "attn_q":     (-20.375,   +20.250,   -34.500,  +34.250),
    "attn_v":     (-20.375,   +20.250,   -34.500,  +34.250),
    "conv_pw1":   (-32.500,   +32.250,   -29.625,  +29.375),
    "conv_pw2":   ( -5.8125,   +5.7813,   -6.250,   +6.1875),
    "ffn_down":   (-11.1875,  +11.0625,  -34.250,  +34.000),
    "ffn_down_1": ( -9.4375,   +9.375,   -33.250,  +33.000),
    "ffn_up":     (-12.875,   +12.8125,  -40.000,  +39.750),
    "ffn_up_1":   (-11.1875,  +11.0625,  -26.375,  +26.125),
}

# All blocks other than block 0 receive identity clamps (±FLT_MAX) because
# Ollama's Safetensors converter only embeds calibration data for block 0.
_CLAMP_IDENTITY = float(np.finfo(np.float32).max)

# The ordered list of audio conformer linear names that carry clamp scalars.
# Matches the keys in _AUDIO_CLAMP_BLK0.
_AUDIO_CLAMP_LINEARS: tuple[str, ...] = tuple(_AUDIO_CLAMP_BLK0.keys())


# ---------------------------------------------------------------------------
# Gemma4ModelCore
# ---------------------------------------------------------------------------

class Gemma4ModelCore(BaseModelCore):
    """Full merge plugin for Gemma 4 multimodal models."""

    MODEL_TYPE: str    = "gemma4"
    REQUIRES_BLOB: bool = False
    STATUS: str        = "stable"

    @classmethod
    #@override
    def get_help_info(cls):
        return {
            "description":   "Gemma 4 — all sizes (E2B, E4B, 26B MoE, 31B dense)",
            "requires_blob": False,
            "status":        "stable",
            "extra_options": [
                ("--vision", "Include vision encoder tensors and KV"),
                ("--audio",  "Include audio encoder tensors and KV (E2B/E4B only)"),
            ],
        }

    def __init__(self, arch: str) -> None:
        super().__init__(arch)
        self._llm_has_clamps: bool = False
        # Populated by process_mmproj_tensors(); checked by post_write_tensors()
        # to avoid double-writing clamp scalars that the mmproj already carries.
        self._encoder_tensor_names: set[str] = set()

    # ── CLI extension ────────────────────────────────────────────────────────

    @classmethod
    #@override
    def add_args(cls, parser) -> None:
        g = parser.add_argument_group("Gemma 4 options")
        g.add_argument(
            "--vision",
            action="store_true",
            default=False,
            help="Include vision encoder tensors and KV (gemma4 only). "
                 "At least one of --vision / --audio is required for gemma4.",
        )
        g.add_argument(
            "--audio",
            action="store_true",
            default=False,
            help="Include audio encoder tensors and KV (gemma4 only, E2B/E4B). "
                 "Errors out if the mmproj has no audio tensors.",
        )

    @classmethod
    #@override
    def format_args_summary(cls, args) -> str | None:
        return (
            f"Gemma4 Multimodal functions:\n"
            f"  Vision: {'Enabled' if args.vision else 'Disabled'}\n"
            f"  Audio:  {'Enabled' if args.audio  else 'Disabled'}\n"
        )

    @classmethod
    #@override
    def validate_args(cls, args) -> None:
        if not args.vision and not args.audio:
            sys.exit(
                "ERROR: gemma4 requires at least one of --vision or --audio; "
                "needs something to merge, can be both."
            )

    # ── KV drop set ──────────────────────────────────────────────────────────

    #@override
    def get_kv_drop(self) -> set[str]:
        a = self.arch
        extra: set[str] = {
            # Sourced from llm_fields — drop to prevent passthrough duplicate
            f"{a}.attention.head_count_kv",
            f"{a}.attention.key_length",
            f"{a}.attention.key_length_swa",
            f"{a}.attention.value_length",
            f"{a}.attention.value_length_swa",
            f"{a}.attention.sliding_window",
            f"{a}.attention.sliding_window_pattern",
            f"{a}.attention.shared_kv_layers",
            f"{a}.embedding_length_per_layer_input",
            f"{a}.rope.dimension_count",
            f"{a}.rope.dimension_count_swa",
            f"{a}.rope.freq_base",
            f"{a}.rope.freq_base_swa",
            f"{a}.final_logit_softcapping",
            f"{a}.feed_forward_length",
            # Sourced from mmproj_fields — drop to prevent passthrough duplicate
            f"{a}.vision.feed_forward_length",
            f"{a}.vision.num_channels",
            # Hardcoded constant — drop LLM passthrough copy
            f"{a}.vision.projector.scale_factor",
            # Tokenizer — re-injected from llm_fields with controlled types
            "tokenizer.ggml.eos_token_ids",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.add_eos_token",
            "tokenizer.ggml.add_padding_token",
            "tokenizer.ggml.add_mask_token",
            "tokenizer.ggml.add_unknown_token",
            "tokenizer.ggml.model",
            "tokenizer.ggml.pre",
            "tokenizer.ggml.scores",
            "tokenizer.ggml.token_type",
            # Audio KV — sourced from mmproj_fields, drop LLM copy
            f"{a}.audio.attention.head_count",
            f"{a}.audio.attention.layer_norm_epsilon",
            f"{a}.audio.block_count",
            f"{a}.audio.conv_kernel_size",
            f"{a}.audio.embedding_length",
            f"{a}.audio.feed_forward_length",
        }
        return super().get_kv_drop() | extra

    # ── KV renames ───────────────────────────────────────────────────────────

    #@override
    def get_kv_renames(self) -> dict[str, str]:
        a = self.arch
        return {
            "clip.vision.block_count":                  f"{a}.vision.block_count",
            "clip.vision.embedding_length":             f"{a}.vision.embedding_length",
            "clip.vision.attention.head_count":         f"{a}.vision.attention.head_count",
            "clip.vision.attention.layer_norm_epsilon": f"{a}.vision.attention.layer_norm_epsilon",
            "clip.vision.patch_size":                   f"{a}.vision.patch_size",
            "clip.vision.image_mean":                   f"{a}.vision.image_mean",
            "clip.vision.image_std":                    f"{a}.vision.image_std",
            # Gemma4 retains image_size (unlike Qwen which uses shortest/longest edge)
            "clip.vision.image_size":                   f"{a}.vision.image_size",
        }

    # ── mmproj KV conditional filter ─────────────────────────────────────────

    #@override
    def should_skip_mmproj_kv(
        self, field_name: str, renamed_key: str, args
    ) -> bool:
        a = self.arch
        # Strip vision KV when --vision is not requested
        if not args.vision and (
            field_name.startswith("clip.vision.")
            or renamed_key.startswith(f"{a}.vision.")
        ):
            return True
        # Audio KV from mmproj is injected explicitly in inject_kv; suppress passthrough
        if field_name.startswith("clip.audio."):
            return True
        return False

    # ── KV injection ─────────────────────────────────────────────────────────

    def inject_kv( self, writer: GGUFWriter, ref_fields: dict | None, mmproj_fields: dict, llm_fields: dict, *, args, ) -> None:  # pyright: ignore[reportMissingTypeArgument]
        """
        Inject all architecture-critical KV fields for Gemma 4.

        Sources (in priority order per field):
          llm_fields    — LLM architecture KV (attention, rope, logit cap, tokenizer)
          mmproj_fields — Encoder architecture KV (vision/audio block counts etc.)
          hardcoded     — Constants verified against Ollama convert_gemma4.go
        """
        a = self.arch

        # ── Attention ──────────────────────────────────────────────────────
        # head_count_kv: scalar for 26B/31B (uniform), per-layer array for E2B/E4B
        hckv_fqn = f"{a}.attention.head_count_kv"
        hckv_field = llm_fields[hckv_fqn]
        if hckv_field.types[0] == GGUFValueType.ARRAY:
            writer.add_array(hckv_fqn, _read_array(llm_fields, hckv_fqn))
        else:
            writer.add_uint32(hckv_fqn, int(_read_scalar(llm_fields, hckv_fqn)))

        writer.add_uint32(
            f"{a}.attention.key_length",
            int(_read_scalar(llm_fields, f"{a}.attention.key_length")),
        )
        writer.add_uint32(
            f"{a}.attention.value_length",
            int(_read_scalar(llm_fields, f"{a}.attention.value_length")),
        )

        for k in ("key_length_swa", "value_length_swa"):
            fqn = f"{a}.attention.{k}"
            if fqn in llm_fields:
                writer.add_uint32(fqn, int(_read_scalar(llm_fields, fqn)))

        writer.add_uint32(
            f"{a}.attention.sliding_window",
            int(_read_scalar(llm_fields, f"{a}.attention.sliding_window")),
        )

        swp_fqn = f"{a}.attention.sliding_window_pattern"
        swp = _read_array(llm_fields, swp_fqn)
        writer.add_array(swp_fqn, [bool(x) for x in swp])

        fqn_skv = f"{a}.attention.shared_kv_layers"
        if fqn_skv in llm_fields:
            writer.add_uint32(fqn_skv, int(_read_scalar(llm_fields, fqn_skv)))

        # embedding_length_per_layer_input — top-level key, absent on 26B MoE/31B dense
        fqn_embd = f"{a}.embedding_length_per_layer_input"
        if fqn_embd in llm_fields:
            val = int(_read_scalar(llm_fields, fqn_embd))
            writer.add_uint32(fqn_embd, val)
            print(f"  ✓ injected {fqn_embd} = {val}")
        else:
            print(f"  NOTE: {fqn_embd} absent (26B MoE / 31B dense — expected)")

        # ── RoPE ───────────────────────────────────────────────────────────
        writer.add_uint32(
            f"{a}.rope.dimension_count",
            int(_read_scalar(llm_fields, f"{a}.rope.dimension_count")),
        )
        fqn_dc_swa = f"{a}.rope.dimension_count_swa"
        if fqn_dc_swa in llm_fields:
            writer.add_uint32(fqn_dc_swa, int(_read_scalar(llm_fields, fqn_dc_swa)))

        writer.add_float32(
            f"{a}.rope.freq_base",
            float(_read_scalar(llm_fields, f"{a}.rope.freq_base")),
        )
        fqn_fb_swa = f"{a}.rope.freq_base_swa"
        if fqn_fb_swa in llm_fields:
            writer.add_float32(fqn_fb_swa, float(_read_scalar(llm_fields, fqn_fb_swa)))

        # ── Logit softcapping ───────────────────────────────────────────────
        writer.add_float32(
            f"{a}.final_logit_softcapping",
            float(_read_scalar(llm_fields, f"{a}.final_logit_softcapping")),
        )

        # ── feed_forward_length — scalar or per-layer array (E2B/E4B) ──────
        ffl_fqn = f"{a}.feed_forward_length"
        if ffl_fqn in llm_fields:
            ffl_field = llm_fields[ffl_fqn]
            if ffl_field.types[0] == GGUFValueType.ARRAY:
                writer.add_array(ffl_fqn, [int(x) for x in _read_array(llm_fields, ffl_fqn)])
            else:
                writer.add_uint32(ffl_fqn, int(_read_scalar(llm_fields, ffl_fqn)))

        # ── Vision KV (mmproj + hardcoded) ─────────────────────────────────
        if args.vision:
            # scale_factor: Go converter defaults to 3 when absent
            # clip.vision.pooling_kernel_size is the mmproj source; fall back to 3
            _scale_key = "clip.vision.pooling_kernel_size"
            scale_factor = (
                int(_read_scalar(mmproj_fields, _scale_key))
                if _scale_key in mmproj_fields
                else 3
            )
            writer.add_uint32(f"{a}.vision.projector.scale_factor", scale_factor)

            # feed_forward_length from mmproj
            _vffl_key = "clip.vision.feed_forward_length"
            if _vffl_key in mmproj_fields:
                writer.add_uint32(
                    f"{a}.vision.feed_forward_length",
                    int(_read_scalar(mmproj_fields, _vffl_key)),
                )

            # num_channels: Go converter defaults to 3 when absent
            _vch_key = "clip.vision.num_channels"
            num_channels = (
                int(_read_scalar(mmproj_fields, _vch_key))
                if _vch_key in mmproj_fields
                else 3
            )
            writer.add_uint32(f"{a}.vision.num_channels", num_channels)

        # ── Audio KV (mmproj, conditional — E2B/E4B only) ──────────────────
        if args.audio:
            _audio_kv: list[tuple[str, str, str]] = [
                # (mmproj clip.audio.* key,  output fqn,  type)
                ("clip.audio.attention.head_count",
                 f"{a}.audio.attention.head_count",        "uint32"),
                ("clip.audio.attention.layer_norm_epsilon",
                 f"{a}.audio.attention.layer_norm_epsilon","float32"),
                ("clip.audio.block_count",
                 f"{a}.audio.block_count",                 "uint32"),
                ("clip.audio.conv_kernel_size",
                 f"{a}.audio.conv_kernel_size",            "uint32"),
                ("clip.audio.embedding_length",
                 f"{a}.audio.embedding_length",            "uint32"),
                ("clip.audio.feed_forward_length",
                 f"{a}.audio.feed_forward_length",         "uint32"),
            ]
            for src_key, out_fqn, vtype in _audio_kv:
                if src_key not in mmproj_fields:
                    # layer_norm_epsilon defaults to 1e-6 (Go: if eps==0 { eps=1e-6 })
                    if src_key == "clip.audio.attention.layer_norm_epsilon":
                        writer.add_float32(out_fqn, 1e-6)
                        print(f"  NOTE: {src_key} absent — using default 1e-6")
                    else:
                        print(f"  WARNING: audio KV '{src_key}' not in mmproj — skipping")
                    continue
                val = _read_scalar(mmproj_fields, src_key)
                if vtype == "uint32":
                    writer.add_uint32(out_fqn, int(val))
                else:
                    writer.add_float32(out_fqn, float(val))

        # ── Tokenizer (llm_fields passthrough, controlled) ──────────────────
        # add_eos_token: Go converter always forces False
        writer.add_bool("tokenizer.ggml.add_eos_token", False)

        # eos_token_ids — prefer array form, fall back to scalar eos_token_id
        if "tokenizer.ggml.eos_token_ids" in llm_fields:
            eos_ids = _read_array(llm_fields, "tokenizer.ggml.eos_token_ids")
            writer.add_array("tokenizer.ggml.eos_token_ids", [int(x) for x in eos_ids])
        elif "tokenizer.ggml.eos_token_id" in llm_fields:
            eos_id = int(_read_scalar(llm_fields, "tokenizer.ggml.eos_token_id"))
            writer.add_array("tokenizer.ggml.eos_token_ids", [eos_id])

        if "tokenizer.ggml.eos_token_id" in llm_fields:
            writer.add_uint32(
                "tokenizer.ggml.eos_token_id",
                int(_read_scalar(llm_fields, "tokenizer.ggml.eos_token_id")),
            )

        for str_key in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
            if str_key in llm_fields:
                f = llm_fields[str_key]
                writer.add_string(str_key, bytes(f.parts[f.data[0]]).decode("utf-8"))

        for arr_key in ("tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
            if arr_key in llm_fields:
                copy_field(writer, llm_fields[arr_key], name=arr_key)

        for bool_key in (
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_padding_token",
            "tokenizer.ggml.add_mask_token",
            "tokenizer.ggml.add_unknown_token",
        ):
            if bool_key in llm_fields:
                writer.add_bool(bool_key, bool(_read_scalar(llm_fields, bool_key)))

        if "tokenizer.ggml.bos_token_id" in llm_fields:
            writer.add_uint32(
                "tokenizer.ggml.bos_token_id",
                int(_read_scalar(llm_fields, "tokenizer.ggml.bos_token_id")),
            )

        # ── General metadata ────────────────────────────────────────────────
        # general.parameter_count does not exist in Ollama Gemma4 blobs — omit.
        _ft = (
            int(_read_scalar(llm_fields, "general.file_type"))
            if "general.file_type" in llm_fields
            else 32
        )
        writer.add_uint32("general.file_type", _ft)

    # ── LLM tensor renames — Gemma 4 needs none ──────────────────────────────

    #def get_llm_renames(self, ref_fields=None, llm_fields=None) -> dict[str, str]:
    #    return {}

    # ── Pre-scan LLM for clamp scalars ───────────────────────────────────────

    #@override
    def prepare_llm(self, llm) -> None:
        """
        Check whether the LLM GGUF already contains clamp scalar tensors.
        If it does, skip the synthesis step in post_write_tensors().
        """
        self._llm_has_clamps = any(
            any(t.name.endswith(s) for s in _CLAMP_SUFFIXES)
            for t in llm.tensors
            if t.name.startswith(("a.", "v."))
        )
        if self._llm_has_clamps:
            print("  NOTE: LLM already contains clamp scalar tensors")

    # ── LLM tensor drop filter ────────────────────────────────────────────────

    #@override
    def should_drop_llm_tensor(
        self, name: str, *, args,  # pyright: ignore[reportMissingTypeArgument]
    ) -> bool:
        if name.startswith(("a.", "v.")):
            # Keep clamp scalars if the LLM already supplies them
            if self._llm_has_clamps and any(name.endswith(s) for s in _CLAMP_SUFFIXES):
                return False
            return True
        return False

    # ── mmproj tensor processing ──────────────────────────────────────────────

    #@override
    def process_mmproj_tensors(self, mmproj, args) -> dict:  # pyright: ignore[reportMissingTypeArgument]
        """
        Gemma 4 mmproj passthrough with:
        - Modality filtering via --vision / --audio flags
        - Audio tensor renames (ggml-org llama.cpp names → Ollama blob names)
        - Validation that audio tensors exist when --audio is requested
        """
        mmproj_names = {t.name for t in mmproj.tensors}
        has_audio = any(
            n.startswith("a.") or n.startswith("mm.a.")
            for n in mmproj_names
        )

        if args.audio and not has_audio:
            sys.exit(
                "ERROR: --audio was specified but the mmproj contains no audio "
                "tensors (a.* / mm.a.*). Use an E2B or E4B mmproj, or drop --audio."
            )
        if has_audio and not args.audio:
            print("  NOTE: mmproj has audio tensors but --audio not set; audio will be stripped.")

        encoder_tensors: dict = {}
        skipped_audio = skipped_vision = renamed_count = 0

        for t in mmproj.tensors:
            is_audio    = t.name.startswith("a.") or t.name.startswith("mm.a.")
            is_vision   = (
                t.name.startswith("v.")
                or t.name == "mm.input_projection.weight"
                or t.name == "rope_freqs.weight"
            )
            is_perlayer = t.name.startswith("per_layer_")  # E2B/E4B audio extras

            if is_audio:
                if args.audio:
                    final_name = _gemma4_audio_rename(t.name)
                    if final_name != t.name:
                        print(f"  tensor rename: {t.name} → {final_name}")
                        renamed_count += 1
                    encoder_tensors[final_name] = t
                else:
                    skipped_audio += 1

            elif is_vision:
                if args.vision:
                    encoder_tensors[t.name] = t
                else:
                    skipped_vision += 1

            elif is_perlayer:
                # per_layer_* belong to the audio pathway (E2B/E4B)
                if args.audio:
                    encoder_tensors[t.name] = t
                else:
                    skipped_audio += 1

            else:
                # Unknown tensor — include to be safe
                encoder_tensors[t.name] = t

        print(f"  Encoder tensors included : {len(encoder_tensors)}")
        if renamed_count:
            print(f"  Audio tensors renamed    : {renamed_count}")
        if skipped_audio:
            print(f"  Audio tensors stripped   : {skipped_audio}")
        if skipped_vision:
            print(f"  Vision tensors stripped  : {skipped_vision}")

        self._encoder_tensor_names = set(encoder_tensors.keys())

        mmproj_clamps = [
            n for n in self._encoder_tensor_names
            if any(n.endswith(s) for s in _CLAMP_SUFFIXES)
        ]
        if mmproj_clamps:
            print(f"  Clamp scalars in mmproj  : {len(mmproj_clamps)} (synthesis will skip these)")

        return encoder_tensors

    # ── Post-write: synthesise audio clamp scalars ───────────────────────────

    #@override
    def post_write_tensors(self, writer: GGUFWriter, ref, args) -> None:
        """
        Gemma4ClippableLinear layers need 1-element F32 clamp scalar tensors
        (name + ".input_min" / ".input_max" / ".output_min" / ".output_max").
        clip.cpp reads them into clamp_info_map; without them, clamping falls
        back to ±FLT_MAX for every layer, degrading encoder quality.

        Strategy (no blob required):
          • Block 0:  inject calibrated constants from _AUDIO_CLAMP_BLK0,
                      verified identical across E2B and E4B Ollama blobs.
          • All other blocks: inject ±FLT_MAX identity clamps.
                      Ollama's Safetensors pipeline only embeds block-0
                      calibration data, so identity is the correct default.

        Skip entirely if:
          • --audio is not set (no audio encoder in output)
          • The LLM GGUF already supplied clamp scalars
          • The mmproj already covered all clamp scalars
        """
        if not args.audio:
            return

        if self._llm_has_clamps:
            print("\nSkipping clamp scalar synthesis — LLM already has them.")
            return

        # Count audio blocks from encoder tensor names
        block_indices = {
            int(m.group(1))
            for n in self._encoder_tensor_names
            if (m := re.match(r"a\.blk\.(\d+)\.", n))
        }
        if not block_indices:
            print("  WARNING: no a.blk.* tensors found — cannot synthesise audio clamps.")
            return

        num_blocks = max(block_indices) + 1

        # Find which clamp scalars the mmproj already wrote
        already_written = {
            n for n in self._encoder_tensor_names
            if any(n.endswith(s) for s in _CLAMP_SUFFIXES) and n.startswith("a.")
        }

        total = num_blocks * len(_AUDIO_CLAMP_LINEARS) * 4
        skipped = 0
        print(
            f"\n  Synthesising audio clamp scalars "
            f"({num_blocks} blocks × {len(_AUDIO_CLAMP_LINEARS)} linears × 4)..."
        )

        for blk in range(num_blocks):
            for linear in _AUDIO_CLAMP_LINEARS:
                if blk == 0 and linear in _AUDIO_CLAMP_BLK0:
                    imin, imax, omin, omax = _AUDIO_CLAMP_BLK0[linear]
                else:
                    imin = omin = -_CLAMP_IDENTITY
                    imax = omax =  _CLAMP_IDENTITY

                for sfx, val in (
                    (".input_min",  imin),
                    (".input_max",  imax),
                    (".output_min", omin),
                    (".output_max", omax),
                ):
                    tname = f"a.blk.{blk}.{linear}{sfx}"
                    if tname in already_written:
                        skipped += 1
                        continue
                    write_tensor(
                        writer, tname,
                        np.array([val], dtype=np.float32),
                        GGMLQuantizationType.F32,
                        [1],
                    )

        written = total - skipped
        if skipped:
            print(f"  Skipped {skipped} clamp scalar(s) already present from mmproj.")
        print(f"  Audio clamp scalars written: {written}")


# ---------------------------------------------------------------------------
# Module-level audio tensor rename helper
# ---------------------------------------------------------------------------

def _gemma4_audio_rename(name: str) -> str:
    """
    Rename a single mmproj audio tensor from its llama.cpp (ggml-org) name
    to the corresponding Ollama blob name.

    Authoritative sources cross-referenced:
      • llama.cpp gguf-py/gguf/constants.py         — GGUF output names
      • llama.cpp gguf-py/gguf/tensor_mapping.py    — HF → GGUF mapping
      • llama.cpp convert_hf_to_gguf.py (Gemma4)   — modify_tensors
      • Ollama convert/convert_gemma4.go            — Replacements()
      • Ollama model/gemma4/model_audio.go          — Go struct gguf tags

    Block-level norm mapping (3 norms per conformer block):
      HF name          llama.cpp mmproj   Ollama blob       Go struct field
      norm_pre_attn    attn_pre_norm       ln1               AttnPreNorm
      norm_post_attn   attn_post_norm      ln2               AttnPostNorm
      norm_out         ln2 (*)             layer_pre_norm    Norm (block final)

    (*) llama.cpp maps norm_out via A_ENC_OUTPUT_NORM → "a.blk.{bid}.ln2".
        Ollama calls the same tensor "layer_pre_norm" via Replacements().

    Each tensor receives at most ONE rename lookup — no chaining.
    """
    # Per-block renames (a.blk.N.xxx → a.blk.N.yyy)
    m = re.match(r"(a\.blk\.\d+\.)(.*)", name)
    if m:
        prefix, suffix = m.group(1), m.group(2)
        blk_renames = {
            "attn_pre_norm.weight":  "ln1.weight",             # AttnPreNorm  gguf:"ln1"
            "attn_post_norm.weight": "ln2.weight",             # AttnPostNorm gguf:"ln2"
            "ln2.weight":            "layer_pre_norm.weight",  # Norm (block) gguf:"layer_pre_norm"
            "attn_k_rel.weight":     "linear_pos.weight",      # LinearPos    gguf:"linear_pos"
        }
        if suffix in blk_renames:
            return prefix + blk_renames[suffix]

    # Top-level projector renames
    # mmproj "a.pre_encode.out"      = audio output projection → Ollama "mm.a.fc"
    # mmproj "a.input_projection"    = SSCP input proj linear  → Ollama "a.pre_encode.out"
    # mmproj "mm.a.input_projection" = audio embedding proj    → unchanged
    proj_renames = {
        "a.pre_encode.out.weight":   "mm.a.fc.weight",
        "a.pre_encode.out.bias":     "mm.a.fc.bias",
        "a.input_projection.weight": "a.pre_encode.out.weight",
    }
    if name in proj_renames:
        return proj_renames[name]

    return name