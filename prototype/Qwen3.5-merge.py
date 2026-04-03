from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType
import numpy as np
from tqdm import tqdm

# === FILE PATHS ===
LLM_PATH      = "Qwen3.5-4B.Q6_K.gguf"  # Input finetuned text model
MMPROJ_PATH   = "mmproj.gguf"            # Input vision encoder for the finetuned model
OUT_PATH      = "merged_qwen35vl.gguf"   # Output merged model
OFFICIAL_BLOB = r"/var/lib/ollama/blobs/sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490"
# OFFICIAL_BLOB is the known-working Ollama base model. Architecture-critical KV values
# (mrope, vision edges, token IDs, head counts) are read from it instead of hardcoded,
# so the merged output is structurally identical to what Ollama validated.

# === HELPERS ===
def _read_array(fields, key):
    f = fields[key]
    return np.concatenate([f.parts[idx] for idx in f.data]).tolist()

def _read_scalar(fields, key):
    f = fields[key]
    return f.parts[f.data[0]][0]

# === LOAD REFERENCE BLOB ===
print("Loading reference blob...")
_ref = GGUFReader(OFFICIAL_BLOB)
_ref_fields = _ref.fields

OFFICIAL_CHAT_TEMPLATE = _ref_fields["tokenizer.chat_template"].parts[
    _ref_fields["tokenizer.chat_template"].data[0]].tobytes().decode("utf-8")
print(f"  chat template: {len(OFFICIAL_CHAT_TEMPLATE)} chars")

REF_SCORES = None
if "tokenizer.ggml.scores" in _ref_fields:
    f = _ref_fields["tokenizer.ggml.scores"]
    REF_SCORES = np.concatenate([f.parts[idx] for idx in f.data]).tolist()
    print(f"  scores: {len(REF_SCORES)} entries")

# Read architecture-critical values from the official blob.
# These control rope, vision resolution, and token routing — wrong values cause crashes.
REF_MROPE_SECTIONS      = _read_array(_ref_fields, "qwen35.mrope_sections")
REF_ROPE_DIM_SECTIONS   = _read_array(_ref_fields, "qwen35.rope.dimension_sections")
REF_ROPE_MROPE_SECTION  = _read_array(_ref_fields, "qwen35.rope.mrope_section") \
                           if "qwen35.rope.mrope_section" in _ref_fields else REF_MROPE_SECTIONS
REF_KV_HEAD_COUNTS      = _read_array(_ref_fields, "qwen35.attention.head_count_kv")
REF_PARAM_COUNT         = int(_read_scalar(_ref_fields, "general.parameter_count"))
REF_IMAGE_TOKEN_ID      = int(_read_scalar(_ref_fields, "qwen35.image_token_id"))
REF_VIS_START_TOKEN_ID  = int(_read_scalar(_ref_fields, "qwen35.vision_start_token_id"))
REF_VIS_END_TOKEN_ID    = int(_read_scalar(_ref_fields, "qwen35.vision_end_token_id"))
REF_LONGEST_EDGE        = int(_read_scalar(_ref_fields, "qwen35.vision.longest_edge"))
REF_SHORTEST_EDGE       = int(_read_scalar(_ref_fields, "qwen35.vision.shortest_edge"))
REF_FILE_TYPE           = int(_read_scalar(_ref_fields, "general.file_type"))

# Layer count is derived from the kv_head_counts array length, not hardcoded
LLM_NUM_LAYERS = len(REF_KV_HEAD_COUNTS)

print(f"  mrope_sections:  {REF_MROPE_SECTIONS}")
print(f"  kv_head_counts:  {REF_KV_HEAD_COUNTS}  ({LLM_NUM_LAYERS} layers)")
print(f"  image_token_id:  {REF_IMAGE_TOKEN_ID}")
print(f"  parameter_count: {REF_PARAM_COUNT}")
print(f"  longest_edge:    {REF_LONGEST_EDGE}")

del _ref, _ref_fields

# === LOAD MMPROJ ===
# Vision dimensions are read from the mmproj KV to avoid hardcoding per model size.
mmproj = GGUFReader(MMPROJ_PATH)
_mf = mmproj.fields

VIT_HIDDEN = int(_mf["clip.vision.embedding_length"].parts[
    _mf["clip.vision.embedding_length"].data[0]][0])
VIT_DEPTH  = int(_mf["clip.vision.block_count"].parts[
    _mf["clip.vision.block_count"].data[0]][0])

print(f"\nVision encoder: hidden={VIT_HIDDEN}, depth={VIT_DEPTH}")

# === QKV SPLITTING ===
# Ollama expects separate Q, K, V tensors. The mmproj stores them as a single fused tensor.

def split_qkv_weight(qkv_weight_tensor):
    data = np.asarray(qkv_weight_tensor.data)
    if qkv_weight_tensor.tensor_type in (GGMLQuantizationType.BF16, GGMLQuantizationType.F16):
        data = data.view(np.uint16)
    qkv = data.reshape(3, VIT_HIDDEN, VIT_HIDDEN)
    return qkv[0].copy(), qkv[1].copy(), qkv[2].copy()

def split_qkv_bias(qkv_bias_data, tensor_type=None):
    data = np.asarray(qkv_bias_data)
    qkv = data.reshape(3, VIT_HIDDEN)
    return qkv[0].copy(), qkv[1].copy(), qkv[2].copy()

# === DEEPSTACK LAYER INDICES ===
# Deepstack layers are vision encoder blocks whose outputs are reused for deep stacking.
# Indices are read from the mmproj if present, otherwise evenly spaced across VIT_DEPTH.
if "clip.vision.is_deepstack_layers" in _mf:
    _ds_key = next((k for k in _mf if "deepstack" in k and "index" in k), None)
    if _ds_key:
        DS_IDXS = [int(x) for x in _read_array(_mf, _ds_key)]
    else:
        step = VIT_DEPTH // 3
        DS_IDXS = [step, step * 2, step * 3]
else:
    step = VIT_DEPTH // 3
    DS_IDXS = [step, step * 2, step * 3]
print(f"Deepstack indices: {DS_IDXS}")

# === TENSOR RENAMING ===
# Maps mmproj tensor names → Ollama's expected fused-model naming convention.
RENAMES = {
    "v.patch_embd.weight":    "v.patch_embed.weight",
    "v.patch_embd.bias":      "v.patch_embed.bias",
    "v.position_embd.weight": "v.pos_embed.weight",
    "mm.0.weight":            "v.merger.linear_fc1.weight",
    "mm.0.bias":              "v.merger.linear_fc1.bias",
    "mm.2.weight":            "v.merger.linear_fc2.weight",
    "mm.2.bias":              "v.merger.linear_fc2.bias",
    "v.post_ln.weight":       "v.merger.norm.weight",
    "v.post_ln.bias":         "v.merger.norm.bias",
}

for merger_idx, vit_layer_idx in enumerate(DS_IDXS):
    for suffix in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                   "norm.weight", "norm.bias"):
        src_field = suffix.split(".")[0]
        src_attr  = suffix.split(".")[1]
        src_key   = f"v.deepstack.{vit_layer_idx}.{suffix}"
        dst_key   = f"v.deepstack_merger.{merger_idx}.linear_{src_field}.{src_attr}" \
                    if src_field in ("fc1", "fc2") \
                    else f"v.deepstack_merger.{merger_idx}.norm.{src_attr}"
        RENAMES[src_key] = dst_key

for i in range(VIT_DEPTH):
    RENAMES[f"v.blk.{i}.ffn_up.bias"]     = f"v.blk.{i}.mlp.linear_fc1.bias"
    RENAMES[f"v.blk.{i}.ffn_up.weight"]   = f"v.blk.{i}.mlp.linear_fc1.weight"
    RENAMES[f"v.blk.{i}.ffn_down.bias"]   = f"v.blk.{i}.mlp.linear_fc2.bias"
    RENAMES[f"v.blk.{i}.ffn_down.weight"] = f"v.blk.{i}.mlp.linear_fc2.weight"
    RENAMES[f"v.blk.{i}.ln1.bias"]        = f"v.blk.{i}.norm1.bias"
    RENAMES[f"v.blk.{i}.ln1.weight"]      = f"v.blk.{i}.norm1.weight"
    RENAMES[f"v.blk.{i}.ln2.bias"]        = f"v.blk.{i}.norm2.bias"
    RENAMES[f"v.blk.{i}.ln2.weight"]      = f"v.blk.{i}.norm2.weight"

# === TENSOR DROP ===
TENSOR_DROP          = {"v.patch_embd.weight.1"}
TENSOR_DROP_PATTERNS = []

def should_drop_tensor(name):
    if name in TENSOR_DROP:
        return True
    return any(pat in name for pat in TENSOR_DROP_PATTERNS)

# === KV FILTERING ===
# KV_DROP prevents duplicate or incompatible keys from the source files.
# Values in this set are either re-injected manually below or intentionally omitted.
KV_DROP = {
    "tokenizer.chat_template",
    "qwen35.attention.head_count_kv",
    "clip.has_vision_encoder", "clip.projector_type", "clip.use_gelu",
    "clip.vision.feed_forward_length", "clip.vision.image_size",
    "clip.vision.is_deepstack_layers", "clip.vision.projection_dim",
    "qwen35.image_token_id", "qwen35.vision_start_token_id",
    "qwen35.vision_end_token_id", "qwen35.vision.longest_edge",
    "qwen35.vision.shortest_edge", "qwen35.mrope_sections",
    "qwen35.rope.dimension_sections", "qwen35.rope.mrope_section",
    "tokenizer.ggml.padding_token_id", "general.parameter_count",
    "tokenizer.ggml.add_eos_token", "tokenizer.ggml.add_padding_token",
    "tokenizer.ggml.eos_token_ids", "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.bos_token_id",
    "general.name", "general.type", "general.size_label", "general.license",
    "general.tags", "general.languages", "general.base_model.count",
    "general.base_model.0.name", "general.base_model.0.organization",
    "general.base_model.0.repo_url", "general.sampling.top_k",
    "general.sampling.top_p", "general.file_type",
    "general.quantization_version",
}

# Maps clip.vision.* keys to the qwen35.vision.* namespace Ollama expects
KV_RENAMES = {
    "clip.vision.block_count":                  "qwen35.vision.block_count",
    "clip.vision.embedding_length":             "qwen35.vision.embedding_length",
    "clip.vision.attention.head_count":         "qwen35.vision.attention.head_count",
    "clip.vision.attention.layer_norm_epsilon": "qwen35.vision.attention.layer_norm_epsilon",
    "clip.vision.patch_size":                   "qwen35.vision.patch_size",
    "clip.vision.spatial_merge_size":           "qwen35.vision.spatial_merge_size",
    "clip.vision.image_mean":                   "qwen35.vision.image_mean",
    "clip.vision.image_std":                    "qwen35.vision.image_std",
}

# === CORE FUNCTIONS ===
FLOAT_TYPES = {
    GGMLQuantizationType.F16, GGMLQuantizationType.F32,
    GGMLQuantizationType.BF16, GGMLQuantizationType.F64,
}

def copy_field(writer, field, name=None):
    if name is None:
        name = field.name
    if name in ("general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
        return
    vtype = field.types[0]
    if   vtype == GGUFValueType.UINT8:   writer.add_uint8(name,   field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.INT8:    writer.add_int8(name,    field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.UINT16:  writer.add_uint16(name,  field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.INT16:   writer.add_int16(name,   field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.UINT32:  writer.add_uint32(name,  field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.INT32:   writer.add_int32(name,   field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.FLOAT32: writer.add_float32(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.UINT64:  writer.add_uint64(name,  field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.INT64:   writer.add_int64(name,   field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.FLOAT64: writer.add_float64(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.BOOL:    writer.add_bool(name,    bool(field.parts[field.data[0]][0]))
    elif vtype == GGUFValueType.STRING:
        writer.add_string(name, str(bytes(field.parts[field.data[0]]), encoding="utf-8"))
    elif vtype == GGUFValueType.ARRAY:
        if not field.data:
            print(f"  NOTE: skipping empty array '{name}'")
            return
        etype = field.types[1]
        if etype == GGUFValueType.STRING:
            arr = [str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data]
        else:
            arr = np.concatenate([field.parts[idx] for idx in field.data]).tolist()
        writer.add_array(name, arr)
    else:
        print(f"  WARNING: skipping unknown type {vtype} for '{name}'")

def write_tensor(writer, name, data, dtype, shape=None):
    if shape is None and hasattr(data, "shape"):
        shape = [int(x) for x in data.shape]
    if dtype in FLOAT_TYPES:
        writer.add_tensor(name, data, raw_shape=shape, raw_dtype=dtype)
    else:
        writer.add_tensor(name, data, raw_dtype=dtype)

# === MMPROJ TENSOR PROCESSING ===
vision_tensors = {}
qkv_tensors    = {}

for t in mmproj.tensors:
    if should_drop_tensor(t.name):
        print(f"  Dropping: {t.name}")
        continue
    if ".attn_qkv." in t.name:
        qkv_tensors[t.name] = t
        continue
    final_name = RENAMES.get(t.name, t.name)
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
        q_b, k_b, v_b = split_qkv_bias(qkv_tensor.data, qkv_tensor.tensor_type)
        vision_tensors[f"v.blk.{blk_idx}.attn_q.bias"] = (q_b, qkv_tensor.tensor_type, None)
        vision_tensors[f"v.blk.{blk_idx}.attn_k.bias"] = (k_b, qkv_tensor.tensor_type, None)
        vision_tensors[f"v.blk.{blk_idx}.attn_v.bias"] = (v_b, qkv_tensor.tensor_type, None)
    elif "weight" in qkv_name:
        q_w, k_w, v_w = split_qkv_weight(qkv_tensor)
        vision_tensors[f"v.blk.{blk_idx}.attn_q.weight"] = (q_w, qkv_tensor.tensor_type, None)
        vision_tensors[f"v.blk.{blk_idx}.attn_k.weight"] = (k_w, qkv_tensor.tensor_type, None)
        vision_tensors[f"v.blk.{blk_idx}.attn_v.weight"] = (v_w, qkv_tensor.tensor_type, None)
print(f"Vision tensors after QKV split: {len(vision_tensors)}")

# Stack dual patch embeddings into the combined tensor Ollama expects.
# Qwen3.5 uses two patch embedding kernels (temporal); they must be interleaved as one tensor.
t0 = vision_tensors.pop("v.patch_embed.weight")
t1 = next(t for t in mmproj.tensors if t.name == "v.patch_embd.weight.1")
d0 = np.asarray(t0.data)
d1 = np.asarray(t1.data)
patch_embed_out = VIT_HIDDEN * 3
combined = np.stack([d0, d1], axis=2).reshape(patch_embed_out, 2, 16, 16).astype(np.float16)
vision_tensors["v.patch_embed.weight"] = (combined, GGMLQuantizationType.F16, None)
print(f"  Stacked patch_embed: {list(d0.shape)} × 2 → {list(combined.shape)}")

# === PREPARE INPUT MODELS ===
llm = GGUFReader(LLM_PATH)

# Quantization version comes from the input text model, not the reference blob,
# because the finetuner controls what quantization scheme was applied.
LLM_QUANT_VERSION = int(_read_scalar(llm.fields, "general.quantization_version")) \
                    if "general.quantization_version" in llm.fields else 2

# Use the finetuned model's chat template when present — finetuners often modify it.
# Fall back to the official blob's template if the LLM doesn't carry one.
if "tokenizer.chat_template" in llm.fields:
    f = llm.fields["tokenizer.chat_template"]
    LLM_CHAT_TEMPLATE = f.parts[f.data[0]].tobytes().decode("utf-8")
    print(f"  Using finetuned chat template ({len(LLM_CHAT_TEMPLATE)} chars)")
else:
    LLM_CHAT_TEMPLATE = OFFICIAL_CHAT_TEMPLATE
    print(f"  WARNING: LLM has no chat template — falling back to official blob")

writer = GGUFWriter(OUT_PATH, arch="qwen35")

# === KV: LLM PASSTHROUGH ===
print("\nCopying LLM KV metadata...")
for field in llm.fields.values():
    if field.name in KV_DROP:
        continue
    renamed = KV_RENAMES.get(field.name, field.name)
    copy_field(writer, field, name=renamed)

# === KV: MMPROJ PASSTHROUGH ===
print("Copying vision KV metadata from mmproj...")
llm_keys = set(llm.fields.keys())
for field in mmproj.fields.values():
    if field.name in llm_keys or field.name in KV_DROP:
        continue
    if field.name in ("general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
        continue
    renamed = KV_RENAMES.get(field.name, field.name)
    if renamed != field.name:
        print(f"  KV rename: {field.name} -> {renamed}")
    copy_field(writer, field, name=renamed)

# === KV: MANUAL INJECTIONS ===
# Fields dropped from the source files are re-injected here with verified values.
# Sources are annotated per-line: blob = official reference, llm = finetuned text model.
print("Injecting KV keys...")

writer.add_uint32("qwen35.vision.num_channels",              3)
writer.add_uint32("qwen35.vision.temporal_patch_size",       2)
writer.add_uint32("qwen35.vision.num_positional_embeddings", 2304)
writer.add_float32("qwen35.vision.rope.freq_base",           10000.0)
writer.add_uint32("qwen35.vision.longest_edge",              REF_LONGEST_EDGE)        # blob
writer.add_uint32("qwen35.vision.shortest_edge",             REF_SHORTEST_EDGE)       # blob

writer.add_uint32("qwen35.image_token_id",                   REF_IMAGE_TOKEN_ID)      # blob
writer.add_uint32("qwen35.vision_start_token_id",            REF_VIS_START_TOKEN_ID)  # blob
writer.add_uint32("qwen35.vision_end_token_id",              REF_VIS_END_TOKEN_ID)    # blob

writer.add_array("qwen35.mrope_sections",          REF_MROPE_SECTIONS)                # blob
writer.add_array("qwen35.rope.dimension_sections", REF_ROPE_DIM_SECTIONS)             # blob
writer.add_array("qwen35.rope.mrope_section",      REF_ROPE_MROPE_SECTION)            # blob
writer.add_bool("qwen35.rope.mrope_interleaved",   True)
writer.add_bool("qwen35.ssm.v_head_reordered",     True)

writer.add_array("qwen35.attention.head_count_kv", REF_KV_HEAD_COUNTS)               # blob

writer.add_uint32("tokenizer.ggml.padding_token_id",  248044)
writer.add_bool("tokenizer.ggml.add_eos_token",       False)
writer.add_bool("tokenizer.ggml.add_padding_token",   False)
writer.add_array("tokenizer.ggml.eos_token_ids",      [151645, 151643])
if REF_SCORES is not None:
    writer.add_array("tokenizer.ggml.scores",         REF_SCORES)

writer.add_uint64("general.parameter_count",       REF_PARAM_COUNT)                  # blob
writer.add_uint32("general.file_type",             REF_FILE_TYPE)                    # blob
writer.add_uint32("general.quantization_version",  LLM_QUANT_VERSION)                # llm

writer.add_string("tokenizer.chat_template", LLM_CHAT_TEMPLATE)
print(f"  ✓ chat template ({len(LLM_CHAT_TEMPLATE)} chars)")

# === TENSORS: LLM ===
# Rename ssm_dt.bias → ssm_dt for each layer (Ollama drops the .bias suffix here)
LLM_RENAMES = {f"blk.{i}.ssm_dt.bias": f"blk.{i}.ssm_dt" for i in range(LLM_NUM_LAYERS)}

dropped_tensors = []
for t in tqdm(llm.tensors, desc="Writing LLM tensors", unit="tensor", leave=True):
    if should_drop_tensor(t.name):
        dropped_tensors.append(t.name)
        continue
    final_name = LLM_RENAMES.get(t.name, t.name)
    data = np.asarray(t.data)
    if t.tensor_type == GGMLQuantizationType.BF16:
        data = data.view(np.uint16)
    shape = [int(x) for x in t.shape[::-1]] if t.tensor_type in FLOAT_TYPES else None
    write_tensor(writer, final_name, data, t.tensor_type, shape)

if dropped_tensors:
    print(f"  Dropped {len(dropped_tensors)} LLM tensors: {dropped_tensors[:5]}")

# === TENSORS: VISION ===
for final_name, t_or_tuple in tqdm(vision_tensors.items(), desc="Writing vision tensors", unit="tensor", leave=True):
    if hasattr(t_or_tuple, "tensor_type"):
        t     = t_or_tuple
        data  = np.asarray(t.data)
        dtype = t.tensor_type
        if dtype == GGMLQuantizationType.BF16:
            data = data.view(np.uint16)
        shape = [int(x) for x in t.shape[::-1]] if dtype in FLOAT_TYPES else None
    else:
        data, dtype, shape = t_or_tuple

    if final_name == "v.pos_embed.weight":
        data  = np.asarray(data).astype(np.float16)
        dtype = GGMLQuantizationType.F16
        shape = [2304, VIT_HIDDEN]

    write_tensor(writer, final_name, data, dtype, shape)

# === FINALIZE ===
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=True)
writer.close()

written_llm = len(llm.tensors) - len(dropped_tensors)
print(f"\nOutput: {OUT_PATH}")
print(f"  LLM tensors:    {written_llm}  ({len(dropped_tensors)} dropped)")
print(f"  Vision tensors: {len(vision_tensors)}")
print(f"  Total:          {written_llm + len(vision_tensors)}")
print("Done.")
