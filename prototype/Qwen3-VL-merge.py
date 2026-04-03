from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType
import numpy as np

# === TENSOR TRANSFORMATIONS FOR OLLAMA COMPATIBILITY ===
def split_qkv_weight(qkv_weight_tensor):
    data = np.asarray(qkv_weight_tensor.data)
    if qkv_weight_tensor.tensor_type in (GGMLQuantizationType.BF16, GGMLQuantizationType.F16):
        data = data.view(np.uint16)   # raw bytes → 2-byte elements
    qkv = data.reshape(3, 1152, 1152)
    return qkv[0].copy(), qkv[1].copy(), qkv[2].copy()

def split_qkv_bias(qkv_bias_data, tensor_type=None):
    data = np.asarray(qkv_bias_data)
    # F32 bias — direct reshape, no view needed
    qkv = data.reshape(3, 1152)
    return qkv[0].copy(), qkv[1].copy(), qkv[2].copy()


# === TENSOR RENAMING RULES ===
RENAMES = {
    "v.patch_embd.weight":        "v.patch_embed.weight",
    "v.patch_embd.bias":          "v.patch_embed.bias",
    "v.position_embd.weight":     "v.pos_embed.weight",
    "mm.0.weight":                "v.merger.linear_fc1.weight",
    "mm.0.bias":                  "v.merger.linear_fc1.bias",
    "mm.2.weight":                "v.merger.linear_fc2.weight",
    "mm.2.bias":                  "v.merger.linear_fc2.bias",
    "v.post_ln.weight":           "v.merger.norm.weight",
    "v.post_ln.bias":             "v.merger.norm.bias",
    # Deepstack merger renames
    "v.deepstack.8.fc1.weight":   "v.deepstack_merger.0.linear_fc1.weight",
    "v.deepstack.8.fc1.bias":     "v.deepstack_merger.0.linear_fc1.bias",
    "v.deepstack.8.fc2.weight":   "v.deepstack_merger.0.linear_fc2.weight",
    "v.deepstack.8.fc2.bias":     "v.deepstack_merger.0.linear_fc2.bias",
    "v.deepstack.8.norm.weight":  "v.deepstack_merger.0.norm.weight",
    "v.deepstack.8.norm.bias":    "v.deepstack_merger.0.norm.bias",
    "v.deepstack.16.fc1.weight":  "v.deepstack_merger.1.linear_fc1.weight",
    "v.deepstack.16.fc1.bias":    "v.deepstack_merger.1.linear_fc1.bias",
    "v.deepstack.16.fc2.weight":  "v.deepstack_merger.1.linear_fc2.weight",
    "v.deepstack.16.fc2.bias":    "v.deepstack_merger.1.linear_fc2.bias",
    "v.deepstack.16.norm.weight": "v.deepstack_merger.1.norm.weight",
    "v.deepstack.16.norm.bias":   "v.deepstack_merger.1.norm.bias",
    "v.deepstack.24.fc1.weight":  "v.deepstack_merger.2.linear_fc1.weight",
    "v.deepstack.24.fc1.bias":    "v.deepstack_merger.2.linear_fc1.bias",
    "v.deepstack.24.fc2.weight":  "v.deepstack_merger.2.linear_fc2.weight",
    "v.deepstack.24.fc2.bias":    "v.deepstack_merger.2.linear_fc2.bias",
    "v.deepstack.24.norm.weight": "v.deepstack_merger.2.norm.weight",
    "v.deepstack.24.norm.bias":   "v.deepstack_merger.2.norm.bias",
}

# Vision block renames (ln -> norm, ffn -> mlp)
for i in range(27):
    RENAMES[f"v.blk.{i}.ffn_up.bias"]     = f"v.blk.{i}.mlp.linear_fc1.bias"
    RENAMES[f"v.blk.{i}.ffn_up.weight"]   = f"v.blk.{i}.mlp.linear_fc1.weight"
    RENAMES[f"v.blk.{i}.ffn_down.bias"]   = f"v.blk.{i}.mlp.linear_fc2.bias"
    RENAMES[f"v.blk.{i}.ffn_down.weight"] = f"v.blk.{i}.mlp.linear_fc2.weight"
    RENAMES[f"v.blk.{i}.ln1.bias"]        = f"v.blk.{i}.norm1.bias"
    RENAMES[f"v.blk.{i}.ln1.weight"]      = f"v.blk.{i}.norm1.weight"
    RENAMES[f"v.blk.{i}.ln2.bias"]        = f"v.blk.{i}.norm2.bias"
    RENAMES[f"v.blk.{i}.ln2.weight"]      = f"v.blk.{i}.norm2.weight"

# === FILTERING RULES ===
DROP = {
    "v.patch_embd.weight.1",
}

KV_DROP = {
    "tokenizer.chat_template", ## we are injecting the official template and discarding the bad one.

    "clip.has_vision_encoder",
    "clip.projector_type",
    "clip.use_gelu",
    # mmproj clip.* keys the blob doesn't use
    "clip.vision.feed_forward_length",
    "clip.vision.image_size", # handled by shortest/longest_edge instead
    "clip.vision.is_deepstack_layers",
    "clip.vision.projection_dim",
    # LLM keys blob doesn't have
    #"qwen3vl.n_deepstack_layers",
    #"qwen3vl.rope.dimension_sections",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.bos_token_id",
    # block general.* from mmproj
    "general.name", "general.type", "general.size_label", "general.license",
    "general.tags", "general.languages", "general.base_model.count",
    "general.base_model.0.name", "general.base_model.0.organization",
    "general.base_model.0.repo_url", "general.sampling.top_k", "general.sampling.top_p",
    "general.file_type",  # injected manually at correct value 15
}

KV_RENAMES = {
    "clip.vision.block_count":                  "qwen3vl.vision.block_count",
    "clip.vision.embedding_length":             "qwen3vl.vision.embedding_length",
    #"clip.vision.feed_forward_length":          "qwen3vl.vision.feed_forward_length",
    "clip.vision.attention.head_count":         "qwen3vl.vision.attention.head_count",
    "clip.vision.attention.layer_norm_epsilon": "qwen3vl.vision.attention.layer_norm_epsilon",
    "clip.vision.patch_size":                   "qwen3vl.vision.patch_size",
    "clip.vision.image_size":                   "qwen3vl.vision.image_size",
    "clip.vision.spatial_merge_size":           "qwen3vl.vision.spatial_merge_size",
    "clip.vision.image_mean":                   "qwen3vl.vision.image_mean",
    "clip.vision.image_std":                    "qwen3vl.vision.image_std",
    # removed: projection_dim and is_deepstack_layers (dropped, not renamed)
}

"""KV_RENAMES = {
    "clip.vision.block_count":                  "qwen3vl.vision.block_count",
    "clip.vision.embedding_length":             "qwen3vl.vision.embedding_length",
    "clip.vision.feed_forward_length":          "qwen3vl.vision.feed_forward_length",
    "clip.vision.attention.head_count":         "qwen3vl.vision.attention.head_count",
    "clip.vision.attention.layer_norm_epsilon": "qwen3vl.vision.attention.layer_norm_epsilon",
    "clip.vision.patch_size":                   "qwen3vl.vision.patch_size",
    "clip.vision.image_size":                   "qwen3vl.vision.image_size",
    "clip.vision.projection_dim":               "qwen3vl.vision.projection_dim",
    "clip.vision.spatial_merge_size":           "qwen3vl.vision.spatial_merge_size",
    "clip.vision.image_mean":                   "qwen3vl.vision.image_mean",
    "clip.vision.image_std":                    "qwen3vl.vision.image_std",
    "clip.vision.is_deepstack_layers":          "qwen3vl.vision.is_deepstack_layers",
}"""

# === CORE FUNCTIONS ===
FLOAT_TYPES = {
    GGMLQuantizationType.F16,
    GGMLQuantizationType.F32,
    GGMLQuantizationType.BF16,
    GGMLQuantizationType.F64,
}

def copy_field(writer, field, name=None):
    """Copy a field to the writer, optionally with a renamed key"""
    if name is None:
        name = field.name
    # Skip special metadata fields
    if name in ('general.architecture', 'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'):
        return

    vtype = field.types[0]
    if vtype == GGUFValueType.UINT8:     writer.add_uint8(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.INT8:    writer.add_int8(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.UINT16:  writer.add_uint16(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.INT16:   writer.add_int16(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.UINT32:  writer.add_uint32(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.INT32:   writer.add_int32(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.FLOAT32: writer.add_float32(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.UINT64:  writer.add_uint64(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.INT64:   writer.add_int64(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.FLOAT64: writer.add_float64(name, field.parts[field.data[0]][0])
    elif vtype == GGUFValueType.BOOL:    writer.add_bool(name, bool(field.parts[field.data[0]][0]))
    elif vtype == GGUFValueType.STRING:  writer.add_string(name, str(bytes(field.parts[field.data[0]]), encoding='utf-8'))
    elif vtype == GGUFValueType.ARRAY:
        etype = field.types[1]
        if etype == GGUFValueType.STRING:
            arr = [str(bytes(field.parts[idx]), encoding='utf-8') for idx in field.data]
        else:
            arr = np.concatenate([field.parts[idx] for idx in field.data]).tolist()
        writer.add_array(name, arr)
    else:
        print(f"  WARNING: skipping {vtype} for '{name}'")

def write_tensor(writer, name, data, dtype, shape=None):
    # shape MUST be numpy-order here (writer reverses when writing) [web:201]
    if shape is None and hasattr(data, "shape"):
        shape = [int(x) for x in data.shape]

    if dtype in FLOAT_TYPES:
        writer.add_tensor(name, data, raw_shape=shape, raw_dtype=dtype)
    else:
        writer.add_tensor(name, data, raw_dtype=dtype)

'''
def write_tensor(writer, name, data, dtype, shape=None):
    """Write tensor with data, dtype, and optional shape override"""
    if shape is None and hasattr(data, 'shape'):
        shape = [int(x) for x in data.shape[::-1]]
    if dtype in FLOAT_TYPES:
        writer.add_tensor(name, data, raw_shape=shape, raw_dtype=dtype)
    else:
        writer.add_tensor(name, data, raw_dtype=dtype)
'''

# === MAIN PROCESSING ===
# Pull official chat template before writing anything
OFFICIAL_BLOB = "/var/lib/ollama/blobs/sha256-ed12a4674d727a74ac4816c906094ea9d3119fbea46ca93288c3ce4ffbe38c55"
_off = GGUFReader(OFFICIAL_BLOB)
_off_tmpl_field = _off.fields["tokenizer.chat_template"]
OFFICIAL_CHAT_TEMPLATE = str(bytes(_off_tmpl_field.parts[_off_tmpl_field.data[0]]), 'utf-8')
print(f"Loaded official chat template ({len(OFFICIAL_CHAT_TEMPLATE)} chars)")
del _off, _off_tmpl_field  # free immediately, we only needed the template

# Load and process vision model tensors
mmproj = GGUFReader("Jan-v2-VL-high-mmproj.gguf")
vision_tensors = {}
qkv_tensors = {}  # Store QKV fused tensors for splitting

for t in mmproj.tensors:
    if t.name in DROP:
        print(f"  Dropping: {t.name}")
        continue

    # Capture QKV fused tensors for splitting
    if '.attn_qkv.' in t.name:
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

# Split QKV tensors into separate Q, K, V
print(f"\nSplitting {len(qkv_tensors)} QKV tensors...")
for qkv_name, qkv_tensor in qkv_tensors.items():
    parts = qkv_name.split('.')
    blk_idx = parts[2]
    if 'bias' in qkv_name:
        q_b, k_b, v_b = split_qkv_bias(qkv_tensor.data, qkv_tensor.tensor_type)
        vision_tensors[f"v.blk.{blk_idx}.attn_q.bias"] = (q_b, qkv_tensor.tensor_type, None)
        vision_tensors[f"v.blk.{blk_idx}.attn_k.bias"] = (k_b, qkv_tensor.tensor_type, None)
        vision_tensors[f"v.blk.{blk_idx}.attn_v.bias"] = (v_b, qkv_tensor.tensor_type, None)
    elif 'weight' in qkv_name:
        q_w, k_w, v_w = split_qkv_weight(qkv_tensor)
        # Change all three from [1152, 1152] to None:
        vision_tensors[f"v.blk.{blk_idx}.attn_q.weight"] = (q_w, qkv_tensor.tensor_type, None)
        vision_tensors[f"v.blk.{blk_idx}.attn_k.weight"] = (k_w, qkv_tensor.tensor_type, None)
        vision_tensors[f"v.blk.{blk_idx}.attn_v.weight"] = (v_w, qkv_tensor.tensor_type, None)


print(f"Vision tensors after processing: {len(vision_tensors)}")

# Stack dual patch_embed weights into Ollama's format
t0 = vision_tensors.pop("v.patch_embed.weight")
t1 = next(t for t in mmproj.tensors if t.name == "v.patch_embd.weight.1")

# numpy shapes are already [1152, 3, 16, 16] — temporal slices of original [1152, 3, 2, 16, 16]
d0 = np.asarray(t0.data)  # [1152, 3, 16, 16]
d1 = np.asarray(t1.data)  # [1152, 3, 16, 16]

# Reconstruct original PyTorch weight [1152, 3, 2, 16, 16]
combined_pt = np.stack([d0, d1], axis=2)  # [1152, 3, 2, 16, 16]

# Apply official converter's reshape: [1152*3, 2, 16, 16] = [3456, 2, 16, 16]
combined = combined_pt.reshape(3456, 2, 16, 16).astype(np.float16)

# Write with explicit GGUF shape [16, 16, 2, 3456] (numpy shape reversed)
vision_tensors["v.patch_embed.weight"] = (combined, GGMLQuantizationType.F16, None ) # [16, 16, 2, 3456])

print(f"  Stacked patch_embed: {list(d0.shape)} × 2 → {list(combined.shape)} (GGUF: [16, 16, 2, 3456])")

# === OUTPUT GENERATION ===
llm = GGUFReader("Jan-v2-VL-high.gguf") #qwen3vl-llm.gguf")
writer = GGUFWriter("jan-v2-vl-merged.gguf", arch="qwen3vl")

# Copy LLM metadata
print("\nCopying KV metadata...")
for field in llm.fields.values():
    if field.name in KV_DROP:
        continue
    renamed = KV_RENAMES.get(field.name, field.name)
    copy_field(writer, field, name=renamed)

# Copy vision metadata with renaming
print("Copying vision KV metadata from mmproj...")
llm_keys = set(llm.fields.keys())
for field in mmproj.fields.values():
    if field.name in llm_keys or field.name in KV_DROP:
        continue
    if field.name in ('general.architecture', 'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'):
        continue
    renamed = KV_RENAMES.get(field.name, field.name)
    if renamed != field.name:
        print(f"  KV rename: {field.name} -> {renamed}")
    else:
        print(f"  vision KV: {field.name}")
    copy_field(writer, field, name=renamed)

# Inject missing vision KV keys
print("Injecting missing vision KV keys...")
writer.add_uint32("qwen3vl.vision.num_channels", 3)
writer.add_uint32("qwen3vl.vision.temporal_patch_size", 2)
writer.add_uint32("qwen3vl.vision.num_positional_embeddings", 2304)
writer.add_float32("qwen3vl.vision.rope.freq_base", 10000.0)
#writer.add_uint32("qwen3vl.vision.shortest_edge", 65536)
#writer.add_uint32("qwen3vl.vision.longest_edge", 16777216)
writer.add_array("qwen3vl.mrope_sections", [24, 20, 20])
writer.add_array("qwen3vl.vision.deepstack_visual_indexes", [8, 16, 24])
writer.add_uint64("general.parameter_count", 8767123696)
writer.add_bool("tokenizer.ggml.add_eos_token", False)
writer.add_bool("tokenizer.ggml.add_padding_token", False)
writer.add_array("tokenizer.ggml.eos_token_ids", [151645, 151643])
writer.add_uint32("general.file_type", 32)  # Fix file_type: blob=15 (F16 mixed)

# Use Jan's native image pixel bounds from mmproj, not official blob's
writer.add_uint32("qwen3vl.vision.shortest_edge", 65536)     # Jan native
#writer.add_uint32("qwen3vl.vision.longest_edge",  16777216)  # Jan native

# Transplant official chat template (Jan's strips <think>\n)
writer.add_string("tokenizer.chat_template", OFFICIAL_CHAT_TEMPLATE)
print(f"✓ Injected official chat template")

# writer.add_uint32("qwen3vl.n_deepstack_layers", 3)

# Write LLM tensors
for t in llm.tensors:
    data = np.asarray(t.data)
    if t.tensor_type == GGMLQuantizationType.BF16:
        data = data.view(np.uint16)
    # Always use t.shape (GGUF-order) reversed to numpy-order for write_tensor
    if t.tensor_type in FLOAT_TYPES or t.tensor_type == GGMLQuantizationType.BF16:
        shape = [int(x) for x in t.shape[::-1]]
    else:
        shape = None
    write_tensor(writer, t.name, data, t.tensor_type, shape)

# Write vision tensors
for final_name, t_or_tuple in vision_tensors.items():
    if hasattr(t_or_tuple, "tensor_type"):
        t = t_or_tuple
        data = np.asarray(t.data)
        dtype = t.tensor_type

        if dtype == GGMLQuantizationType.BF16:
            data = data.view(np.uint16)

        # ALWAYS derive shape from t.shape, not data.shape (data is often 1D)
        shape = [int(x) for x in t.shape[::-1]] if dtype in FLOAT_TYPES else None

    else:
        data, dtype, shape = t_or_tuple
        # shape here is already numpy-order if you set it; otherwise None is fine

    if final_name == "v.pos_embed.weight":
        data  = np.asarray(data).astype(np.float16)
        dtype = GGMLQuantizationType.F16
        shape = [2304, 1152] # as nummpy flip flops # [1152, 2304]  # numpy-order; writer flips to GGUF [2304,1152] [web:201]

    write_tensor(writer, final_name, data, dtype, shape)



# Finalize output
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("Done.")
