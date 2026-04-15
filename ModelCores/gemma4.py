"""
ModelCores/gemma4.py
====================
ModelCore plugin for Gemma 4 (all sizes: E2B, E4B, 26B MoE, 31B dense).

Gemma 4 specifics handled here
-------------------------------
• CLI flags      --vision / --audio to select which encoder(s) to include
• KV drop        Extended set; most arch-critical KV is re-injected from blob
• KV renames     Gemma4-specific subset (no spatial_merge_size, uses image_size)
• inject_kv      Sources all arch-critical values from the official Ollama blob
• mmproj tensors Passthrough + audio tensor renames (ggml-org → Ollama naming)
• Clamp scalars  Transplanted from blob if not already present in LLM tensors

What this plugin does NOT do (handled by base / engine)
--------------------------------------------------------
• QKV splitting       (Gemma4 mmproj does not fuse QKV)
• Deepstack handling  (Gemma4 has no deepstack vision layers)
• Patch-embed stacking (Gemma4 uses a standard single patch embed)
• LLM tensor renames   (Gemma4 needs none)
"""

from __future__ import annotations

import re
import sys

import numpy as np
from tqdm import tqdm
from gguf import GGUFWriter, GGUFValueType, GGMLQuantizationType

from .base import (
    BaseModelCore,
    FLOAT_TYPES,
    _read_array,
    _read_scalar,
    copy_field,
    write_tensor,
)

# Activation-clamp scalar tensor suffixes (Gemma4ClippableLinear)
_CLAMP_SUFFIXES = (".input_min", ".input_max", ".output_min", ".output_max")


class Gemma4ModelCore(BaseModelCore):
    """Full merge plugin for Gemma 4 multimodal models."""

    MODEL_TYPE = "gemma4"
    REQUIRES_BLOB = True
    STATUS        = "stable"

    @classmethod
    def get_help_info(cls) -> dict:
        return {
            "description":   "Gemma 4 — all sizes (E2B, E4B, 26B MoE, 31B dense)",
            "requires_blob": True,
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

    # CLI extension
    @classmethod
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
    def format_args_summary(cls, args) -> str | None:
        return f"""Gemma4 Multimodal functions:
  Vision: {"Enabled" if args.vision else "Disabled"}
  Audio: {"Enabled" if args.audio else "Disabled"}
        """

    # Argument validation
    @classmethod
    def validate_args(cls, args) -> None:
        if not args.vision and not args.audio:
            sys.exit("ERROR: gemma4 requires at least one of --vision or --audio; needs something to merge it, can be both.")

    # KV drop set
    def get_kv_drop(self) -> set[str]:
        a = self.arch
        extra: set[str] = {
            # Re-injected from blob
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
            f"{a}.vision.projector.scale_factor",
            f"{a}.vision.feed_forward_length",
            f"{a}.vision.num_channels",
            # Tokenizer fields re-injected from blob
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
            # Audio KV — re-injected conditionally from blob
            f"{a}.audio.attention.head_count",
            f"{a}.audio.attention.layer_norm_epsilon",
            f"{a}.audio.block_count",
            f"{a}.audio.conv_kernel_size",
            f"{a}.audio.embedding_length",
            f"{a}.audio.feed_forward_length",
            # Misc
            "general.parameter_count",
        }
        return super().get_kv_drop() | extra

    # KV renames — Gemma4 uses a different subset than Qwen
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

    # mmproj KV conditional filter
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
        # Audio KV from mmproj is always handled by inject_kv; suppress here
        if field_name.startswith("clip.audio."):
            return True
        return False

    # KV injection
    def inject_kv(
        self,
        writer: GGUFWriter,
        ref_fields: dict,
        mmproj_fields: dict,
        llm_fields: dict,
        *,
        args,
    ) -> None:
        """
        Inject all architecture-critical KV fields for Gemma 4.
        Every value is sourced from the official Ollama blob so the merged
        output is structurally identical to what Ollama has validated.
        """
        a = self.arch
        ref = ref_fields  # shorter alias

        # ── Attention (blob) ──────────────────────────────────────────
        # head_count_kv may be a scalar (E2B uniform=1) or per-layer array
        hckv_field = ref[f"{a}.attention.head_count_kv"]
        if hckv_field.types[0] == GGUFValueType.ARRAY:
            writer.add_array(
                f"{a}.attention.head_count_kv",
                _read_array(ref, f"{a}.attention.head_count_kv"),
            )
        else:
            writer.add_uint32(
                f"{a}.attention.head_count_kv",
                int(_read_scalar(ref, f"{a}.attention.head_count_kv")),
            )

        writer.add_uint32(
            f"{a}.attention.key_length",
            int(_read_scalar(ref, f"{a}.attention.key_length")),
        )
        writer.add_uint32(
            f"{a}.attention.value_length",
            int(_read_scalar(ref, f"{a}.attention.value_length")),
        )

        for k in ("key_length_swa", "value_length_swa"):
            fqn = f"{a}.attention.{k}"
            if fqn in ref:
                writer.add_uint32(fqn, int(_read_scalar(ref, fqn)))

        writer.add_uint32(
            f"{a}.attention.sliding_window",
            int(_read_scalar(ref, f"{a}.attention.sliding_window")),
        )

        swp = _read_array(ref, f"{a}.attention.sliding_window_pattern")
        writer.add_array(
            f"{a}.attention.sliding_window_pattern",
            [bool(x) for x in swp],
        )

        fqn_skv = f"{a}.attention.shared_kv_layers"
        if fqn_skv in ref:
            writer.add_uint32(fqn_skv, int(_read_scalar(ref, fqn_skv)))

        # embedding_length_per_layer_input is a top-level key (not under .attention.)
        fqn_embd = f"{a}.embedding_length_per_layer_input"
        if fqn_embd in ref:
            val = int(_read_scalar(ref, fqn_embd))
            writer.add_uint32(fqn_embd, val)
            print(f"  ✓ injected {fqn_embd} = {val}")
        else:
            print(f"  NOTE: {fqn_embd} not in blob (26B MoE/31B dense — expected)")

        # ── RoPE (blob) ───────────────────────────────────────────────
        writer.add_uint32(
            f"{a}.rope.dimension_count",
            int(_read_scalar(ref, f"{a}.rope.dimension_count")),
        )
        if f"{a}.rope.dimension_count_swa" in ref:
            writer.add_uint32(
                f"{a}.rope.dimension_count_swa",
                int(_read_scalar(ref, f"{a}.rope.dimension_count_swa")),
            )
        writer.add_float32(
            f"{a}.rope.freq_base",
            float(_read_scalar(ref, f"{a}.rope.freq_base")),
        )
        if f"{a}.rope.freq_base_swa" in ref:
            writer.add_float32(
                f"{a}.rope.freq_base_swa",
                float(_read_scalar(ref, f"{a}.rope.freq_base_swa")),
            )

        # ── Logit softcapping (blob) ──────────────────────────────────
        writer.add_float32(
            f"{a}.final_logit_softcapping",
            float(_read_scalar(ref, f"{a}.final_logit_softcapping")),
        )

        # ── Vision KV (blob, conditional) ────────────────────────────
        if args.vision:
            writer.add_uint32(
                f"{a}.vision.projector.scale_factor",
                int(_read_scalar(ref, f"{a}.vision.projector.scale_factor")),
            )
            if f"{a}.vision.feed_forward_length" in ref:
                writer.add_uint32(
                    f"{a}.vision.feed_forward_length",
                    int(_read_scalar(ref, f"{a}.vision.feed_forward_length")),
                )
            if f"{a}.vision.num_channels" in ref:
                writer.add_uint32(
                    f"{a}.vision.num_channels",
                    int(_read_scalar(ref, f"{a}.vision.num_channels")),
                )

        # ── Audio KV (blob, conditional — E2B/E4B only) ──────────────
        if args.audio:
            _audio_kv = [
                (f"{a}.audio.attention.head_count",       "uint32"),
                (f"{a}.audio.attention.layer_norm_epsilon","float32"),
                (f"{a}.audio.block_count",                "uint32"),
                (f"{a}.audio.conv_kernel_size",           "uint32"),
                (f"{a}.audio.embedding_length",           "uint32"),
                (f"{a}.audio.feed_forward_length",        "uint32"),
            ]
            for fqn, vtype in _audio_kv:
                if fqn not in ref:
                    print(f"  WARNING: audio KV '{fqn}' not in blob — skipping")
                    continue
                val = _read_scalar(ref, fqn)
                if vtype == "uint32":
                    writer.add_uint32(fqn, int(val))
                else:
                    writer.add_float32(fqn, float(val))

            # feed_forward_length as per-layer array (E2B/E4B)
            ffl_key = f"{a}.feed_forward_length"
            if ffl_key in ref:
                ffl_field = ref[ffl_key]
                if ffl_field.types[0] == GGUFValueType.ARRAY:
                    writer.add_array(ffl_key, [int(x) for x in _read_array(ref, ffl_key)])
                else:
                    writer.add_uint32(ffl_key, int(_read_scalar(ref, ffl_key)))

        # ── Tokenizer (blob) ─────────────────────────────────────────
        eos_ids = _read_array(ref, "tokenizer.ggml.eos_token_ids")
        writer.add_array("tokenizer.ggml.eos_token_ids", [int(x) for x in eos_ids])
        writer.add_bool("tokenizer.ggml.add_eos_token", False)

        for str_key in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
            if str_key in ref:
                val = bytes(ref[str_key].parts[ref[str_key].data[0]]).decode("utf-8")
                writer.add_string(str_key, val)

        if "tokenizer.ggml.eos_token_id" in ref:
            writer.add_uint32(
                "tokenizer.ggml.eos_token_id",
                int(_read_scalar(ref, "tokenizer.ggml.eos_token_id")),
            )

        for arr_key in ("tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
            if arr_key in ref:
                copy_field(writer, ref[arr_key], name=arr_key)

        for bool_key in (
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_padding_token",
            "tokenizer.ggml.add_mask_token",
            "tokenizer.ggml.add_unknown_token",
        ):
            if bool_key in ref:
                writer.add_bool(bool_key, bool(_read_scalar(ref, bool_key)))

        if "tokenizer.ggml.bos_token_id" in ref:
            writer.add_uint32(
                "tokenizer.ggml.bos_token_id",
                int(_read_scalar(ref, "tokenizer.ggml.bos_token_id")),
            )

        # ── General metadata ──────────────────────────────────────────
        writer.add_uint64(
            "general.parameter_count",
            int(_read_scalar(ref, "general.parameter_count")),
        )
        _ft = (
            int(_read_scalar(llm_fields, "general.file_type"))
            if "general.file_type" in llm_fields
            else 32
        )
        writer.add_uint32("general.file_type", _ft)

    # LLM tensor renames — Gemma 4 needs none
    def get_llm_renames(self, ref_fields=None) -> dict[str, str]:
        return {}

    # Pre-scan LLM for clamp scalars
    def prepare_llm(self, llm) -> None:
        """
        Check whether the LLM GGUF already contains clamp scalar tensors.
        If it does, we skip the blob transplant step (post_write_tensors).
        """
        self._llm_has_clamps = any(
            any(t.name.endswith(s) for s in _CLAMP_SUFFIXES)
            for t in llm.tensors
            if t.name.startswith(("a.", "v."))
        )
        if self._llm_has_clamps:
            print("  NOTE: LLM already contains clamp scalar tensors")

    # LLM tensor drop filter
    def should_drop_llm_tensor(
        self, name: str, *, args, encoder_tensors: dict
    ) -> bool:
        if name.startswith(("a.", "v.")):
            # Keep clamp scalars if the LLM already supplies them
            if self._llm_has_clamps and any(name.endswith(s) for s in _CLAMP_SUFFIXES):
                return False
            return True
        return False

    # mmproj tensor processing
    def process_mmproj_tensors(self, mmproj, args) -> dict:
        """
        Gemma 4 mmproj passthrough with:
        - Modality filtering via --vision / --audio flags
        - Audio tensor renames (ggml-org llama.cpp names → Ollama blob names)
        - Validation that audio tensors exist when --audio is requested
        """
        # Detect audio tensor presence before filtering
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
            is_audio   = t.name.startswith("a.") or t.name.startswith("mm.a.")
            is_vision  = (
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

        # Cache the final names so post_write_tensors() can skip any
        # clamp scalars that the mmproj already supplied.
        self._encoder_tensor_names = set(encoder_tensors.keys())

        # Report how many clamp tensors the mmproj carries (diagnostic)
        mmproj_clamps = [
            n for n in self._encoder_tensor_names
            if any(n.endswith(s) for s in _CLAMP_SUFFIXES)
        ]
        if mmproj_clamps:
            print(f"  Clamp scalars in mmproj  : {len(mmproj_clamps)} (blob transplant will skip these)")

        return encoder_tensors

    # Step 12: Post-write — transplant clamp scalars from blob
    def post_write_tensors(self, writer: GGUFWriter, ref, args) -> None:
        """
        Gemma4ClippableLinear layers store learned activation clamp bounds as
        1-element F32 scalar tensors.  For a.* and v.* tensors these scalars
        live in the BLOB's tensor section, not in the mmproj.
        Only mm.* clamp scalars reside in the mmproj.

        clip.cpp reads them via name + ".input_min" etc. and stores them in
        clamp_info_map. Without them, clamping defaults to ±FLT_MAX, degrading
        encoder quality.
        """
        if ref is None:
            return

        if self._llm_has_clamps:
            print("\nSkipping clamp scalar transplant — LLM already has them.")
            return

        clamp_tensors = [
            t for t in ref.tensors
            if any(t.name.endswith(s) for s in _CLAMP_SUFFIXES)
            and (t.name.startswith("a.") or t.name.startswith("v."))
        ]

        if not clamp_tensors:
            print("  NOTE: no clamp scalar tensors found in blob (unexpected for Gemma4)")
            return

        # Filter to only the clamps that are NOT already in encoder_tensors.
        # The mmproj can carry a.*/v.* clamp scalars for some model builds;
        # writing them again from the blob causes a duplicate tensor error.
        missing_clamps = [
            t for t in clamp_tensors
            if t.name not in self._encoder_tensor_names
            and (t.name.startswith("a.") and args.audio
                 or t.name.startswith("v.") and args.vision)
        ]
        already_have = len(clamp_tensors) - len(missing_clamps)
        if already_have:
            print(f"  Skipping {already_have} clamp tensor(s) already written from mmproj")

        if not missing_clamps:
            print("  All clamp scalars covered — no blob transplant needed.")
            return

        print(f"\nCopying {len(missing_clamps)} clamp scalar tensors from blob...")
        for t in tqdm(missing_clamps, desc="Clamp scalars", unit="tensor", leave=True):
            data  = np.asarray(t.data)
            dtype = t.tensor_type  # always F32
            shape = [int(x) for x in t.shape[::-1]] if dtype in FLOAT_TYPES else None
            write_tensor(writer, t.name, data, dtype, shape)

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

    IMPORTANT: blk_renames contains a chain (attn_post_norm→ln2, ln2→layer_pre_norm).
    Each tensor is renamed with a single lookup — there is no chaining.
    """
    # Per-block renames (a.blk.N.xxx → a.blk.N.yyy)
    m = re.match(r"(a\.blk\.\d+\.)(.*)", name)
    if m:
        prefix, suffix = m.group(1), m.group(2)
        blk_renames = {
            "attn_pre_norm.weight":  "ln1.weight",            # AttnPreNorm   gguf:"ln1"
            "attn_post_norm.weight": "ln2.weight",            # AttnPostNorm  gguf:"ln2"
            "ln2.weight":            "layer_pre_norm.weight", # Norm (block)  gguf:"layer_pre_norm"
            "attn_k_rel.weight":     "linear_pos.weight",     # LinearPos     gguf:"linear_pos.weight"
        }
        if suffix in blk_renames:
            return prefix + blk_renames[suffix]

    # Top-level projector renames
    # mmproj "a.pre_encode.out"     = audio output projection  → Ollama "mm.a.fc"
    # mmproj "a.input_projection"   = SSCP input proj linear   → Ollama "a.pre_encode.out"
    # mmproj "mm.a.input_projection"= audio embedding proj     → unchanged
    proj_renames = {
        "a.pre_encode.out.weight":  "mm.a.fc.weight",
        "a.pre_encode.out.bias":    "mm.a.fc.bias",
        "a.input_projection.weight":"a.pre_encode.out.weight",
    }
    if name in proj_renames:
        return proj_renames[name]

    return name
