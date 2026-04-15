#!/usr/bin/env python3
"""
OllamaGGUFMerge.py
==================
Modular Engine for merging a finetuned text LLM + a multimodal projector (mmproj)
into a single Ollama-compatible GGUF.

Model-specific logic lives in ModelCores/<model>.py.  New models are discovered
automatically — drop a file that subclasses BaseModelCore in that directory and
it will appear as a valid --model-type choice the next time this script runs.

Pass -h or --help to see discovered models and all options.
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

from .ModelCores import discover_models, load_model_core
from .ModelCores.base import (
    FLOAT_TYPES, SKIP_META, copy_field, write_tensor, _read_scalar,
    STATUS_STABLE, STATUS_STUB, STATUS_EXPERIMENTAL,
)


# ---------------------------------------------------------------------------
# Custom help
# ---------------------------------------------------------------------------

# Badge strings shown next to each model in the --help model list.
_STATUS_BADGE = {
    STATUS_STABLE:       "  [stable]      ",
    STATUS_STUB:         "  [stub]        ",
    STATUS_EXPERIMENTAL: "  [experimental]",
}

_BLOB_BADGE = {
    True:  "blob required",
    False: "blob optional",
}

def print_models(model_registry: dict) -> None:  # pyright: ignore[reportMissingTypeArgument]
    """Print a formatted table of all discovered model plugins and exit."""
    print(f"\n{'MODEL TYPE':<16} {'STATUS':<14} {'BLOB':<10} DESCRIPTION")
    print("─" * 72)
    for mtype in sorted(model_registry):
        cls  = model_registry[mtype]
        info = cls.get_help_info()
        status   = _STATUS_BADGE.get(info["status"], info["status"])
        blob_tag = "required" if info["requires_blob"] else "optional"
        desc     = info["description"] or ""
        print(f"{mtype:<16} {status:<14} {blob_tag:<10} {desc}")
    print()
    sys.exit(0)

def _print_custom_help(model_registry: dict, model_type: str | None = None) -> None:  # pyright: ignore[reportMissingTypeArgument]
    """
    Print the custom help screen and exit.

    If model_type is given, also show a focused block for that specific model.
    """


    # ── Build the model list block ─────────────────────────────────────────
    model_lines: list[str] = []
    specific_options: list[tuple[str, list[tuple[str, str]]]] = []

    for mtype in sorted(model_registry):
        cls  = model_registry[mtype]
        info = cls.get_help_info()

        badge    = _STATUS_BADGE.get(info["status"], "  [unknown]     ")
        blob_tag = _BLOB_BADGE[info["requires_blob"]]
        desc     = f" — {info['description']}" if info["description"] else ""
        # model_lines.append(f"  {mtype:<16}{badge}  ({blob_tag}){desc}")


        if info["extra_options"]:
            specific_options.append((mtype, info["extra_options"]))

    #models_block = "\n".join(model_lines) if model_registry else "(none found)"

    # ── Build the model-specific options block ─────────────────────────────
    if specific_options:
        spec_sections = []
        for mtype, opts in specific_options:
            lines = [f"  {mtype}:"]
            for flag, desc in opts:
                lines.append(f"    {flag:<22}  {desc}")
            spec_sections.append("\n".join(lines))
        specific_block = "\n\n".join(spec_sections)
    else:
        specific_block = "  (none)"

    # ── Compose full help text ──────────────────────────────────────────────
    script = os.path.basename(sys.argv[0])
    HEADER = "Tsunagi:Ollama-GGUF-Merger"
    DESC = "Modular GGUF Patching platform"

    help_text = f"""
{HEADER} — {DESC}
{'='*(len(HEADER)+len(DESC)+3)}
Merges and patches a finetuned text LLM + a separate mmproj encoder into a single GGUF
that Ollama can load directly as a multimodal model.

Usage:
  python {script} --model-type <type> [options]
  python {script} --model-type <type> --help   (model-focused help)

Available Models: {len(model_registry)}\n\tUse --models to see full list detected models and their compatibilities.

  Status badges:
    [stable]       Fully implemented and tested — use this.
    [stub]         Skeleton only; use the original monolith script for now.
    [experimental] Ported but not yet fully validated in production.

Common Options:
  --model-type,\t-t <type>\tModel architecture (required; see list above)
  --blob,\t-b  <path>\tPath to official Ollama blob for this arch
  \t\t(required for blob-required models, see above)
  --llm,\t-l  <path>\tFinetuned text-model GGUF  (required)
  --mmproj,\t-m  <path>\tMultimodal projector GGUF  (required)
  --output,\t-o  <path>\tOutput path (default: merged.gguf)
  -h, --help\tShow this help and exit
  --models\tList models detected in the modules.

Model-Specific Options:
{specific_block}

Examples:
  # Gemma 4 — vision + audio (E2B/E4B)
  python {script} --model-type gemma4 --blob sha256-... \\
      --llm gemma-4-E2B-it-Q8_0.gguf --mmproj mmproj-gemma-4-E2B-it-f16.gguf \\
      --vision --audio

  # Gemma 4 — vision only (26B MoE / 31B dense, no audio)
  python {script} --model-type gemma4 --blob sha256-... \\
      --llm gemma-4-26B-it-Q4_K_M.gguf --mmproj mmproj-gemma-4-26B-it-f16.gguf \\
      --vision
"""

    # ── If a specific model was requested, append a focused section ─────────
    if model_type and model_type in model_registry:
        cls  = model_registry[model_type]
        info = cls.get_help_info()
        focused = f"""
Model detail: {model_type}
{'─' * (16 + len(model_type))}
  Description  : {info['description'] or '(none)'}
  Blob required: {'yes' if info['requires_blob'] else 'no'}
  Status       : {info['status']}
"""
        if info["extra_options"]:
            focused += "  Extra options:\n"
            for flag, desc in info["extra_options"]:
                focused += f"    {flag:<22}  {desc}\n"
        help_text += focused

    print(help_text.rstrip())
    sys.exit(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(model_registry: dict) -> argparse.Namespace:  # pyright: ignore[reportMissingTypeArgument]
    supported = sorted(model_registry.keys())
    p = argparse.ArgumentParser(
        description="Merge a finetuned LLM + mmproj into a single Ollama-compatible GGUF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Disable built-in -h so our custom help handler takes precedence.
        add_help=False,
    )
    p.add_argument(
        "--model-type", "-t",
        required=False,   # validated manually so we can show help on bad input
        default=None,
        choices=supported,
        metavar="TYPE",
        help=f"Architecture to target. Discovered: {', '.join(supported)}",
    )
    p.add_argument("--blob",   "-b", required=False, default=None,           metavar="PATH")
    p.add_argument("--llm",    "-l", required=False, default=None,           metavar="PATH")
    p.add_argument("--mmproj", "-m", required=False, default=None,           metavar="PATH")
    p.add_argument("--output", "-o", default="merged.gguf",                  metavar="PATH")
    p.add_argument("-h", "--help",   action="store_true", default=False,
                   help="Show custom help and exit")
    p.add_argument("--models", action="store_true", default=False, help="List all available model types and exit.")

    # Let every discovered plugin extend the parser with its own flags.
    for cls in model_registry.values():
        cls.add_args(p)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # Discover plugins first - needed for help display
    print("Discovering model cores...")
    model_registry = discover_models()
    if not model_registry:
        sys.exit("ERROR: No model cores found in ModelCores/")
    print(f"  Registered: {', '.join(sorted(model_registry))}\n")

    # 1. Parse (add_help=False so -h reaches us, not argparse)
    args = _parse_args(model_registry)

    # Custom help gate
    # -h / --help always shows our screen.
    # If --model-type was also provided, the screen gains a focused block.
    if args.help:
        _print_custom_help(model_registry, model_type=args.model_type)
    elif args.models:
        print_models(model_registry)

    # Now enforce required positional args (we disabled argparse's own required check).
    missing = [flag for flag, attr in [("--model-type", "model_type"), ("--llm", "llm"), ("--mmproj", "mmproj")]
               if not getattr(args, attr)]
    if missing:
        # Friendly error pointing toward help instead of a cryptic argparse dump.
        for flag, attr in [("--model-type", "model_type"), ("--llm", "llm"), ("--mmproj", "mmproj")]:
            if not getattr(args, attr):
                print(f"ERROR: {flag} is required.\n")
        print(f"Run\tpython {os.path.basename(sys.argv[0])} --help for usage.")
        sys.exit(1)
    #
    #nstantiate and validate
    #
    core = load_model_core(model_registry, args.model_type)

    if core.REQUIRES_BLOB and not args.blob:
        sys.exit(f"ERROR: --blob is required for model type '{args.model_type}'")

    core.validate_args(args)

    for label, path in [("LLM", args.llm), ("mmproj", args.mmproj)]:
        if not os.path.exists(path):
            sys.exit(f"ERROR: {label} path does not exist: {path}")
    if args.blob and not os.path.exists(args.blob):
        sys.exit(f"ERROR: blob path does not exist: {args.blob}")

    print(f"Model type : {args.model_type}")
    print(f"Blob       : {args.blob}")
    print(f"LLM        : {args.llm}")
    print(f"mmproj     : {args.mmproj}")
    print(f"Output     : {args.output}")
    if summary := core.format_args_summary(args): # For verbal back from module to user based off costom arguments
        print(f"{summary}")

    #
    # Load reference blob 
    #
    if args.blob:
        print("Loading reference blob...")
        ref = GGUFReader(args.blob)
        ref_fields = ref.fields
        if "tokenizer.chat_template" in ref_fields:
            f = ref_fields["tokenizer.chat_template"]
            official_chat_template = bytes(f.parts[f.data[0]]).decode("utf-8")
            print(f"  chat template: {len(official_chat_template)} chars")
        else:
            official_chat_template = None
            print("  chat template: not in blob (will source from LLM)")
    else:
        ref = ref_fields = official_chat_template = None
        print("No blob provided — all KV values will be hardcoded (qwen3vl/qwen3vlmoe).")

    # ── 5. Load mmproj + process encoder tensors ─────────────────────
    print("\nLoading mmproj...")
    mmproj = GGUFReader(args.mmproj)
    encoder_tensors = core.process_mmproj_tensors(mmproj, args)

    # ── 6. Load LLM ──────────────────────────────────────────────────
    print("\nLoading LLM...")
    llm = GGUFReader(args.llm)

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
            "no --blob was provided to fall back to."
        )

    llm_quant_version = (
        int(_read_scalar(llm.fields, "general.quantization_version"))
        if "general.quantization_version" in llm.fields
        else 2
    )

    # ── 7. Open writer ───────────────────────────────────────────────
    writer = GGUFWriter(args.output, arch=args.model_type)
    kv_drop    = core.get_kv_drop()
    kv_renames = core.get_kv_renames()

    # ── 8. Copy LLM KV ───────────────────────────────────────────────
    print("\nCopying LLM KV metadata...")
    for field in llm.fields.values():
        if field.name in kv_drop:
            continue
        copy_field(writer, field, name=kv_renames.get(field.name, field.name))

    # ── 9. Copy mmproj KV ────────────────────────────────────────────
    print("Copying mmproj KV metadata...")
    llm_keys = set(llm.fields.keys())
    for field in mmproj.fields.values():
        if field.name in llm_keys or field.name in kv_drop or field.name in SKIP_META:
            continue
        renamed = kv_renames.get(field.name, field.name)
        if core.should_skip_mmproj_kv(field.name, renamed, args):
            continue
        if renamed != field.name:
            print(f"  KV rename: {field.name} → {renamed}")
        copy_field(writer, field, name=renamed)

    # ── 10. Inject controlled KV fields ─────────────────────────────
    print("Injecting controlled KV fields...")
    core.inject_kv(writer, ref_fields, mmproj.fields, llm.fields, args=args)
    writer.add_string("tokenizer.chat_template", chat_template)
    print(f"  ✓ chat template ({len(chat_template)} chars)")
    writer.add_uint32("general.quantization_version", llm_quant_version)

    # ── 11. Pre-scan LLM ────────────────────────────────────────────
    core.prepare_llm(llm)

    # ── 12. Write LLM tensors ────────────────────────────────────────
    llm_renames     = core.get_llm_renames(ref_fields)
    dropped_tensors: list[str] = []

    for t in tqdm(llm.tensors, desc="Writing LLM tensors", unit="tensor", leave=True):
        final_name = llm_renames.get(t.name, t.name)
        if core.should_drop_llm_tensor(final_name, args=args, encoder_tensors=encoder_tensors):
            dropped_tensors.append(final_name)
            continue
        data = np.asarray(t.data)
        if t.tensor_type == GGMLQuantizationType.BF16:
            data = data.view(np.uint16)
        shape = [int(x) for x in t.shape[::-1]] if t.tensor_type in FLOAT_TYPES else None
        write_tensor(writer, final_name, data, t.tensor_type, shape)

    if dropped_tensors:
        print(f"  Dropped {len(dropped_tensors)} LLM tensors: {dropped_tensors[:5]}"
              + (" ..." if len(dropped_tensors) > 5 else ""))

    # ── 13. Write encoder tensors ────────────────────────────────────
    for final_name, t_or_tuple in tqdm(
        encoder_tensors.items(), desc="Writing encoder tensors", unit="tensor", leave=True
    ):
        if hasattr(t_or_tuple, "tensor_type"):
            t     = t_or_tuple
            data  = np.asarray(t.data)
            dtype = t.tensor_type
            if dtype == GGMLQuantizationType.BF16:
                data = data.view(np.uint16)
            shape = [int(x) for x in t.shape[::-1]] if dtype in FLOAT_TYPES else None
        else:
            data, dtype, shape = t_or_tuple
        write_tensor(writer, final_name, data, dtype, shape)

    # ── 14. Post-write hook ──────────────────────────────────────────
    core.post_write_tensors(writer, ref, args)

    # ── 15. Finalize ─────────────────────────────────────────────────
    print("\nFinalizing output file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    written_llm = len(llm.tensors) - len(dropped_tensors)
    print(f"\nOutput         : {args.output}")
    print(f"  LLM tensors  : {written_llm}"
          + (f" ({len(dropped_tensors)} dropped)" if dropped_tensors else ""))
    print(f"  Enc tensors  : {len(encoder_tensors)}")
    print(f"  Total        : {written_llm + len(encoder_tensors)}")
    print("Done.")


if __name__ == "__main__":
    main()
