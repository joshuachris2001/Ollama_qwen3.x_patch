#!/usr/bin/env python3
"""
app.py — Tsunagi Ollama WebUI
HuggingFace Space interface for the Tsunagi GGUF monolith builder.
Wraps the tsunagi-ollama-bridge pipeline in a Gradio UI.
"""

import os
import shutil
import tempfile
import time
import traceback
from typing import final
import hashlib
import glob
import gradio as gr  # pyright: ignore[reportMissingImports]
import numpy as np
import threading
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from huggingface_hub import hf_hub_url, get_hf_file_metadata, hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError

from tsunagi_ollama_bridge.ModelCores import discover_models, load_model_core  # pyright: ignore[reportMissingImports]
from tsunagi_ollama_bridge.ModelCores.base import (  # pyright: ignore[reportMissingImports]
    FLOAT_TYPES, SKIP_META, copy_field, write_tensor, _read_scalar,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLM_CAP_BYTES    = 23 * 1024 ** 3   # 23 GB
MMPROJ_CAP_BYTES =  1 * 1024 ** 3   #  1 GB

MAX_OPTS = 8

# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

_TMPDIR = tempfile.gettempdir()

def _user_job_dir(session_hash: str) -> str:
    h = hashlib.sha256(session_hash.encode()).hexdigest()[:16]
    return os.path.join(_TMPDIR, f"tsunagi_{h}")

def _cleanup_user_previous(job_dir: str):
    if os.path.isdir(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)

def _start_age_cleanup_worker():
    def _worker():
        while True:
            time.sleep(3600)
            cutoff = time.time() - 86400
            dirs = sorted(glob.glob(os.path.join(_TMPDIR, "tsunagi_*")))[:500]
            for d in dirs:
                try:
                    marker = os.path.join(d, "merged.gguf")
                    if os.path.isfile(marker) and os.path.getmtime(marker) < cutoff:
                        shutil.rmtree(d, ignore_errors=True)
                    elif os.path.isdir(d) and os.path.getmtime(d) < cutoff:
                        shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
    threading.Thread(target=_worker, daemon=True).start()

_start_age_cleanup_worker()

# ---------------------------------------------------------------------------
# Model registry — one pass at startup
# ---------------------------------------------------------------------------
def get_supported_models() -> dict:
    registry = discover_models()
    return {k: v for k, v in registry.items() if not v.REQUIRES_BLOB}

SUPPORTED_MODELS = get_supported_models()
MODEL_CHOICES    = sorted(SUPPORTED_MODELS.keys())

# Pre-scan extra_options for every model once — used by UI and on_submit
MODEL_OPTIONS: dict[str, list[tuple[str, str]]] = {
    k: v.get_help_info().get("extra_options", [])
    for k, v in SUPPORTED_MODELS.items()
}

# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------
def detect_architecture(gguf_path: str) -> str | None:
    try:
        reader = GGUFReader(gguf_path, mode="r")
        for field in reader.fields.values():
            if field.name == "general.architecture":
                return str(bytes(field.parts[-1]), encoding="utf-8").lower()
    except Exception:
        pass
    finally:
        del reader  # pyright: ignore[reportPossiblyUnboundVariable]
    return None

# ---------------------------------------------------------------------------
# HF Hub size validation
# ---------------------------------------------------------------------------
def validate_hub_size(repo_id: str, filename: str, cap: int) -> tuple[bool, int]:
    url  = hf_hub_url(repo_id, filename)
    meta = get_hf_file_metadata(url)
    size = meta.size or 0
    return size <= cap, size

# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------
def resolve_input(upload, repo_id: str, filename: str, cap: int, label: str) -> tuple[str | None, str]:
    cap_gb = cap // 1024 ** 3

    if upload is not None:
        path = upload if isinstance(upload, str) else upload.name
        size = os.path.getsize(path)
        if size > cap:
            return None, f"❌ {label} upload exceeds {cap_gb} GB cap ({size / 1024**3:.1f} GB)."
        return path, f"✓ {label} uploaded ({size / 1024**3:.2f} GB)."

    if repo_id and filename:
        repo_id  = repo_id.strip()
        filename = filename.strip()
        try:
            ok, size = validate_hub_size(repo_id, filename, cap)
            if not ok:
                return None, f"❌ {label} on Hub is {size / 1024**3:.1f} GB — exceeds {cap_gb} GB cap."
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            return path, f"✓ {label} pulled from Hub ({size / 1024**3:.2f} GB)."
        except EntryNotFoundError:
            return None, f"❌ {label}: file not found in {repo_id}."
        except Exception as e:
            return None, f"❌ {label} Hub pull failed: {e}"

    return None, f"⚠ No {label} provided — upload a file or enter a HF repo + filename."

# ---------------------------------------------------------------------------
# Args namespace
# ---------------------------------------------------------------------------
class _Args:
    def __init__(self, model_type, llm, mmproj, output, extra_flags: dict | None = None):
        self.model_type = model_type
        self.llm        = llm
        self.mmproj     = mmproj
        self.blob       = None
        self.output     = output
        for attr, val in (extra_flags or {}).items():
            setattr(self, attr, val)

# ---------------------------------------------------------------------------
# Progress bar / elapsed helpers
# ---------------------------------------------------------------------------
def _progress_bar_html(done: int, total: int, label: str = "") -> str:
    pct = int((done / total) * 100) if total > 0 else 0
    return f"""<div style="margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:var(--body-text-color);margin-bottom:4px">
    <span>{label}</span><span>{done}/{total} ({pct}%)</span>
  </div>
  <div style="background:var(--border-color-primary);border-radius:9999px;height:8px;overflow:hidden">
    <div style="width:{pct}%;height:100%;border-radius:9999px;background:var(--color-accent);transition:width 0.25s ease"></div>
  </div>
</div>"""

def _fmt_elapsed(start: float) -> str:
    elapsed = int(time.time() - start)
    h, rem  = divmod(elapsed, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"

# ---------------------------------------------------------------------------
# Core merge pipeline
# ---------------------------------------------------------------------------
def run_merge_streamed(
    llm_path: str, mmproj_path: str, model_type: str, output_path: str,
    initial_logs: list[str] | None = None, extra_flags: dict | None = None,
):
    logs       = list(initial_logs) if initial_logs else []
    start_time = time.time()

    def log(msg: str):
        logs.append(f"[{_fmt_elapsed(start_time)}] {msg}")

    yield _progress_bar_html(0, 0, "Waiting…"), "\n".join(logs), None

    core = load_model_core(SUPPORTED_MODELS, model_type)
    args = _Args(model_type, llm_path, mmproj_path, output_path, extra_flags)
    core.validate_args(args)

    log("Loading mmproj...")
    yield _progress_bar_html(0, 0, "Loading mmproj…"), "\n".join(logs), None
    mmproj = GGUFReader(args.mmproj)
    encoder_tensors = core.process_mmproj_tensors(mmproj, args)

    log("Loading LLM...")
    yield _progress_bar_html(0, 0, "Loading LLM…"), "\n".join(logs), None
    llm = GGUFReader(args.llm)

    if "tokenizer.chat_template" in llm.fields:
        f = llm.fields["tokenizer.chat_template"]
        chat_template = bytes(f.parts[f.data[0]]).decode("utf-8")
        log(f"  Chat template: {len(chat_template)} chars")
    else:
        log("❌ No chat template in LLM and no blob fallback available.")
        yield _progress_bar_html(0, 0, "Failed"), "\n".join(logs), None
        return

    llm_quant_version = (
        int(_read_scalar(llm.fields, "general.quantization_version"))
        if "general.quantization_version" in llm.fields else 2
    )

    writer     = GGUFWriter(args.output, arch=args.model_type)
    kv_drop    = core.get_kv_drop()
    kv_renames = core.get_kv_renames()

    log("Copying LLM KV metadata...")
    yield _progress_bar_html(0, 0, "Copying KV metadata…"), "\n".join(logs), None
    for field in llm.fields.values():
        if field.name in kv_drop:
            continue
        copy_field(writer, field, name=kv_renames.get(field.name, field.name))

    log("Copying mmproj KV metadata...")
    yield _progress_bar_html(0, 0, "Copying mmproj KV…"), "\n".join(logs), None
    llm_keys = set(llm.fields.keys())
    for field in mmproj.fields.values():
        if field.name in llm_keys or field.name in kv_drop or field.name in SKIP_META:
            continue
        renamed = kv_renames.get(field.name, field.name)
        if core.should_skip_mmproj_kv(field.name, renamed, args):
            continue
        copy_field(writer, field, name=renamed)

    log("Injecting controlled KV fields...")
    core.inject_kv(writer, None, mmproj.fields, llm.fields, args=args)
    del mmproj

    writer.add_string("tokenizer.chat_template", chat_template)
    writer.add_uint32("general.quantization_version", llm_quant_version)

    core.prepare_llm(llm)
    llm_renames = core.get_llm_renames(ref_fields=None, llm_fields=llm.fields)
    dropped: list[str] = []

    log(f"Writing {len(llm.tensors)} LLM tensors...")
    total   = len(llm.tensors) + len(encoder_tensors)
    written = 0
    for t in llm.tensors:
        final_name = llm_renames.get(t.name, t.name)
        if core.should_drop_llm_tensor(final_name, args=args, encoder_tensors=encoder_tensors):
            dropped.append(final_name)
            continue
        data = np.asarray(t.data)
        if t.tensor_type == GGMLQuantizationType.BF16:
            data = data.view(np.uint16)
        shape = [int(x) for x in t.shape[::-1]] if t.tensor_type in FLOAT_TYPES else None
        write_tensor(writer, final_name, data, t.tensor_type, shape)
        written += 1
        if written % 25 == 0:
            yield (
                _progress_bar_html(written, total, f"Writing tensors… {_fmt_elapsed(start_time)}"),
                "\n".join(logs), None,
            )

    log(f"Writing {len(encoder_tensors)} encoder tensors...")
    for final_name, t_or_tuple in encoder_tensors.items():
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
        written += 1
        if written % 25 == 0:
            yield (
                _progress_bar_html(written, total, f"Writing tensors… {_fmt_elapsed(start_time)}"),
                "\n".join(logs), None,
            )

    written_llm = len(llm.tensors) - len(dropped)
    del llm

    core.post_write_tensors(writer, None, args)

    log("Finalizing output GGUF...")
    yield _progress_bar_html(total, total, f"Finalizing… {_fmt_elapsed(start_time)}"), "\n".join(logs), None

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    tensor_count = sum(len(t) for t in writer.tensors)
    log(f"Writing {tensor_count} tensors to file...")
    yielded = total
    for tensors in writer.tensors:
        for name, ti in list(tensors.items()):
            writer.write_tensor_data(ti.tensor)  # pyright: ignore[reportArgumentType]
            yielded += 1
            if yielded % 50 == 0:
                yield (
                    _progress_bar_html(yielded, total + tensor_count, f"Writing tensor data… {_fmt_elapsed(start_time)}"),
                    "\n".join(logs), None,
                )

    writer.close()
    log(
        f"\n✓ Done in {_fmt_elapsed(start_time)}\n"
        f"  LLM tensors   : {written_llm}"
        + (f" ({len(dropped)} dropped)" if dropped else "") + "\n"
        f"  Enc tensors   : {len(encoder_tensors)}\n"
        f"  Total         : {written_llm + len(encoder_tensors)}"
    )
    yield (
        _progress_bar_html(
            total + tensor_count, total + tensor_count,
            f"Complete — {_fmt_elapsed(start_time)} - Syncing model to disk... [Should take a bit]",
        ),
        "\n".join(logs), args.output,
    )

# ---------------------------------------------------------------------------
# Options UI helpers
# ---------------------------------------------------------------------------

_MSG_NO_MODEL   = "*⚠ A model must be selected or detected before options can be shown.*"
_MSG_NO_OPTIONS = "*No model-specific options available for this architecture.*"

def _flag_to_attr(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")

def _options_ui_updates(model_type: str) -> list:
    """
    Returns MAX_OPTS checkbox gr.update()s + 1 notice gr.update().
    Total length always == MAX_OPTS + 1.
    """
    is_real = model_type != "AUTO DETECT" and model_type in SUPPORTED_MODELS
    options = MODEL_OPTIONS.get(model_type, []) if is_real else []

    cb_updates = []
    for i in range(MAX_OPTS):
        if i < len(options):
            flag, desc = options[i]
            cb_updates.append(gr.update(
                label=f"{flag}   —   {desc}",
                visible=True,
                value=flag in ("--vision",),
                interactive=True,
            ))
        else:
            cb_updates.append(gr.update(visible=False, value=False))

    if not is_real:
        notice = gr.update(visible=True, value=_MSG_NO_MODEL)
    elif not options:
        notice = gr.update(visible=True, value=_MSG_NO_OPTIONS)
    else:
        notice = gr.update(visible=False)

    return cb_updates + [notice]

# ---------------------------------------------------------------------------
# detect_btn handler
# Returns ALL outputs in one atomic response — no chaining, no race.
# outputs: [arch_status, model_type_dd, run_btn, cb0…cb7, opts_notice]
# ---------------------------------------------------------------------------
def on_detect(llm_upload, llm_repo, llm_file) -> list:
    path, msg = resolve_input(llm_upload, llm_repo, llm_file, LLM_CAP_BYTES, "LLM")

    if path is None:
        return [msg, "AUTO DETECT", gr.update(interactive=False)] + _options_ui_updates("AUTO DETECT")

    arch = detect_architecture(path)

    if arch is None:
        status = f"{msg} ⚠ Could not read general.architecture from GGUF."
        return [status, "AUTO DETECT", gr.update(interactive=False)] + _options_ui_updates("AUTO DETECT")

    if arch not in MODEL_CHOICES:
        status = f"{msg} ❌ Architecture **{arch}** is not supported."
        return [status, "AUTO DETECT", gr.update(interactive=False)] + _options_ui_updates("AUTO DETECT")

    status = f"{msg} ✓ Architecture detected: **{arch}** — supported."
    return [status, arch, gr.update(interactive=True)] + _options_ui_updates(arch)

# ---------------------------------------------------------------------------
# model_type_dd.change handler
# Returns ALL outputs in one atomic response — same shape as on_detect minus
# the first two (arch_status, model_type_dd are not outputs of this handler).
# outputs: [run_btn, cb0…cb7, opts_notice]
# ---------------------------------------------------------------------------
def on_model_change(model_type: str) -> list:
    is_real = model_type != "AUTO DETECT" and model_type in SUPPORTED_MODELS
    return [gr.update(interactive=is_real)] + _options_ui_updates(model_type)

# ---------------------------------------------------------------------------
# Submit handler
# ---------------------------------------------------------------------------
def on_submit(
    llm_upload, llm_repo, llm_file,
    mmproj_upload, mmproj_repo, mmproj_file,
    model_type,
    hf_push, hf_repo, hf_token,
    *opt_values,
    request: gr.Request
):
    session_hash = getattr(request, "session_hash", None) or "anonymous"
    job_dir      = _user_job_dir(session_hash)
    _cleanup_user_previous(job_dir)
    os.makedirs(job_dir, exist_ok=True)

    logs = []
    def log(msg: str): logs.append(msg)

    yield _progress_bar_html(0, 0, "Starting…"), "", gr.update(interactive=False)

    llm_path, llm_msg = resolve_input(llm_upload, llm_repo, llm_file, LLM_CAP_BYTES, "LLM")
    log(llm_msg)
    if llm_path is None:
        yield _progress_bar_html(0, 0, "Failed"), "\n".join(logs), gr.update()
        return

    if model_type == "AUTO DETECT":
        log("Auto-detecting architecture from LLM GGUF...")
        yield _progress_bar_html(0, 0, "Detecting architecture…"), "\n".join(logs), gr.update()
        arch = detect_architecture(llm_path)
        if arch is None:
            log("❌ Could not read general.architecture — try selecting manually.")
            yield _progress_bar_html(0, 0, "Failed"), "\n".join(logs), gr.update()
            return
        if arch not in SUPPORTED_MODELS:
            log(f"❌ Detected architecture '{arch}' is not supported.\nSupported: {', '.join(MODEL_CHOICES)}")
            yield _progress_bar_html(0, 0, "Failed"), "\n".join(logs), gr.update()
            return
        log(f"✓ Auto-detected architecture: {arch}")
        model_type = arch

    mmproj_path, mm_msg = resolve_input(mmproj_upload, mmproj_repo, mmproj_file, MMPROJ_CAP_BYTES, "mmproj")
    log(mm_msg)
    if mmproj_path is None:
        yield _progress_bar_html(0, 0, "Failed"), "\n".join(logs), gr.update()
        return

    if not model_type:
        log("❌ No model type selected.")
        yield _progress_bar_html(0, 0, "Failed"), "\n".join(logs), gr.update()
        return

    options     = MODEL_OPTIONS.get(model_type, [])
    extra_flags = {
        _flag_to_attr(flag): bool(opt_values[i])
        for i, (flag, _) in enumerate(options)
    }

    last_log_text = "\n".join(logs)
    raw_output    = os.path.join(job_dir, "building.gguf")
    try:
        for bar_html, log_text, file_path in run_merge_streamed(
            llm_path, mmproj_path, model_type, raw_output, logs, extra_flags=extra_flags,
        ):
            last_log_text = log_text
            if file_path is not None:
                raw_output = file_path
            yield bar_html, log_text, gr.update()
    except Exception:
        yield _progress_bar_html(0, 0, "Failed"), f"{last_log_text}\n❌ Merge failed:\n{traceback.format_exc()}", gr.update()
        return

    if raw_output is None:
        yield _progress_bar_html(0, 0, "Failed"), last_log_text, gr.update()
        return

    merged_path = os.path.join(job_dir, "merged.gguf")
    os.rename(raw_output, merged_path)
    output_path = merged_path

    if hf_push and hf_repo and hf_token:
        try:
            upload_file(
                path_or_fileobj=output_path, path_in_repo="merged.gguf",
                repo_id=hf_repo.strip(), token=hf_token.strip(), repo_type="model",
            )
            yield (
                _progress_bar_html(1, 1, "Hub upload complete"),
                f"{last_log_text}\n✓ Uploaded to HF Hub: {hf_repo}/merged.gguf",
                gr.update(value=output_path, interactive=True),
            )
        except Exception as e:
            yield (
                _progress_bar_html(1, 1, "Hub upload failed"),
                f"{last_log_text}\n⚠ Hub upload failed: {e}\nFalling back to direct download.",
                gr.update(value=output_path, interactive=True),
            )
    else:
        yield _progress_bar_html(1, 1, "Ready for download"), last_log_text, gr.update(value=output_path, interactive=True)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Tsunagi — Ollama WebUI") as demo:

    gr.Markdown(
        """
        # 🌉 Tsunagi — Ollama GGUF Monolith Builder [HF Space UI BETA edition]
        Patch a finetuned llama.cpp text LLM (GGUF) + a multimodal projector (mmproj GGUF) into a single
        Ollama-compatible GGUF. Upload files directly or pull from a HuggingFace repo.

        > **Unofficial tool.** Output models are not produced by Ollama directly.

        > ⚠️ Warning: a modelfile is needed to run these resulting models in Ollama;
        > and these Models will not work with llama.cpp as they will be Ollama only.

        > This is a hobby project, so I might be slow on adding models that are not in demand; and I will not work on models that are not officially compatible with Ollama.
        > That would require an even deeper reverse engendering then its worth; and ATM it's not worth it.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### LLM (text model) — max 23 GB")
            llm_upload = gr.File(label="Upload GGUF", file_types=[".gguf"])
            gr.Markdown("*or pull from HuggingFace Hub:*")
            llm_repo   = gr.Textbox(label="Repo ID", placeholder="org/model-name")
            llm_file   = gr.Textbox(label="Filename", placeholder="model.gguf")

        with gr.Column():
            gr.Markdown("### mmproj (vision encoder) — max 1 GB")
            mmproj_upload = gr.File(label="Upload GGUF", file_types=[".gguf"])
            gr.Markdown("*or pull from HuggingFace Hub:*")
            mmproj_repo   = gr.Textbox(label="Repo ID", placeholder="org/model-name")
            mmproj_file   = gr.Textbox(label="Filename", placeholder="mmproj.gguf")

    # ── Architecture detection ──────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            arch_status = gr.Markdown("*Architecture will be detected from the LLM GGUF.*")
        with gr.Column(scale=1):
            detect_btn = gr.Button("🔍 Detect Architecture", variant="secondary")

    # ── Model type ──────────────────────────────────────────────────
    with gr.Row():
        model_type_dd = gr.Dropdown(
            choices=["AUTO DETECT"] + MODEL_CHOICES,
            value="AUTO DETECT",
            label="Model Type",
            info="Auto-filled on detection, or select manually.",
            interactive=True,
        )

    # Supported architectures table — includes Options column from pre-scan
    def _arch_row(k: str) -> str:
        opts     = MODEL_OPTIONS.get(k, [])
        opts_str = ", ".join(f"`{flag}`" for flag, _ in opts) if opts else "—"
        return f"| `{k}` | ✓ | {opts_str} |"

    with gr.Accordion("Supported architectures", open=False):
        gr.Markdown(
            "| Model | Status | Options |\n|---|---|---|\n"
            + "\n".join(_arch_row(k) for k in MODEL_CHOICES)
        )

    # run_btn — starts non-interactive (AUTO DETECT selected)
    run_btn = gr.Button("⚙ Build Monolith", variant="primary", interactive=False)

    # ── Model-specific options ──────────────────────────────────────
    with gr.Row():
        gr.Markdown("**Model-specific options:**")
    with gr.Row():
        opt_checkboxes: list[gr.Checkbox] = []
        for _ in range(MAX_OPTS):
            cb = gr.Checkbox(label="", value=False, visible=False, interactive=True)
            opt_checkboxes.append(cb)

    # Starts visible — correct for AUTO DETECT initial state
    opts_notice = gr.Markdown(value=_MSG_NO_MODEL, visible=True)

    # ── Shared output list used by BOTH handlers ────────────────────
    # [run_btn, cb0…cb7, opts_notice]  — MAX_OPTS + 2 items
    _opts_outputs = [run_btn] + opt_checkboxes + [opts_notice]

    # detect_btn: single atomic response covering every output it touches.
    # Does NOT write model_type_dd from a second step — all outputs resolved
    # in one fn call, returned together, applied by Gradio in one update.
    detect_btn.click(
        fn=on_detect,
        inputs=[llm_upload, llm_repo, llm_file],
        outputs=[arch_status, model_type_dd] + _opts_outputs,
    )

    # model_type_dd.change: handles manual selection.
    # Separate handler, separate outputs — no overlap with detect_btn outputs
    # so there is no possibility of a concurrent write to the same component.
    model_type_dd.input(
        fn=on_model_change,
        inputs=[model_type_dd],
        outputs=_opts_outputs,
    )

    # ── Output options ──────────────────────────────────────────────
    gr.Markdown("### Output")
    with gr.Row():
        with gr.Column():
            hf_push  = gr.Checkbox(label="Push result to HF Hub", value=False)
            hf_repo  = gr.Textbox(label="Destination Repo ID", placeholder="your-username/model-name", visible=False)
            hf_token = gr.Textbox(label="HF Write Token", placeholder="hf_...", type="password", visible=False)
            hf_push.change(
                fn=lambda v: (gr.update(visible=v), gr.update(visible=v)),
                inputs=hf_push,
                outputs=[hf_repo, hf_token],
            )

    progress_html = gr.HTML(value="", visible=True)
    log_box       = gr.Textbox(label="Log", lines=18, interactive=False)

    download_btn = gr.DownloadButton(
        label="⬇️ Download merged.gguf",
        value=None, interactive=False,
        variant="primary", size="lg",
        elem_id="tsunagi-download-btn",
    )

    gr.HTML("""
    <style>
      #tsunagi-download-btn { margin-top: 8px; }
      #tsunagi-download-btn button {
        width: 100%; min-height: 56px;
        font-size: 1.15rem; font-weight: 600; letter-spacing: 0.02em;
      }
    </style>
    """)

    run_btn.click(
        fn=on_submit,
        inputs=[
            llm_upload, llm_repo, llm_file,
            mmproj_upload, mmproj_repo, mmproj_file,
            model_type_dd,
            hf_push, hf_repo, hf_token,
            *opt_checkboxes,
        ],
        outputs=[progress_html, log_box, download_btn],
    )

    gr.Markdown(
        """
        ---
        *Tsunagi is not affiliated with Ollama, Inc. Models produced by this tool may differ
        from models produced by official Ollama tooling.*
        *Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*
        """
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default())