"""
ModelCores/qwen35.py
====================
ModelCore plugins for Qwen3.5-VL dense+SSM hybrid and Qwen3.5-VL-MoE.

Qwen3.5-VL specifics
---------------------
• REQUIRES_BLOB = False
	All architecture-critical values are sourced directly from the text LLM
	GGUF (llm_fields), the mmproj GGUF (mmproj_fields), and a vocabulary
	scan — no Ollama blob is required.

	Sources per field:
	  llm_fields  → attention.head_count_kv, mrope_sections,
					rope.dimension_sections, rope.mrope_section,
					rope.mrope_interleaved, ssm.v_head_reordered,
					general.parameter_count, general.file_type,
					tokenizer.ggml.eos_token_ids, tokenizer.ggml.scores
	  mmproj_fields → vision.deepstack_visual_indexes
					  (derived from clip.vision.is_deepstack_layers)
	  vocab scan  → image_token_id, vision_start_token_id,
					vision_end_token_id  (fallback to known constants)
	  hardcoded   → vision geometry (identical across all model sizes,
					confirmed against 4B/9B/27B/35B-A3B preprocessor configs)

• SSM hybrid layer rename (get_llm_renames)
	The finetuned LLM stores the SSM delta-time bias as "blk.N.ssm_dt.bias"
	but Ollama's llama.cpp backend expects "blk.N.ssm_dt" (no .bias suffix).
	Layer count is read from llm_fields' attention.head_count_kv array.

• rope.mrope_interleaved / ssm.v_head_reordered flags
	Read directly from llm_fields; both are written there by
	convert_qwen3next.go at HuggingFace→GGUF conversion time.

• Deepstack indexes sourced from mmproj
	mmproj carries clip.vision.is_deepstack_layers as a bool array.
	Indexes where True are collected into vision.deepstack_visual_indexes.
	MoE models produce an empty list (no deepstack tensors in mmproj).

• mmproj pipeline
	Full Qwen pipeline inherited from QwenBaseModelCore:
	QKV split + deepstack renames (indexes scanned from mmproj tensors)
	+ patch-embed stacking.
"""

from __future__ import annotations

from gguf import GGUFWriter

from .base import _read_array, _read_scalar
from .qwen_base import QwenBaseModelCore  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Known Hardcoded Constants — vision geometry identical across all Qwen3.5 model sizes
# (verified against 4B / 9B / 27B / 35B-A3B preprocessor_config.json)
# ---------------------------------------------------------------------------

_VISION_LONGEST_EDGE  = 16_777_216
_VISION_SHORTEST_EDGE =     65_536

# Fallback token IDs — used only when the vocab scan fails.
# Stable across all official Qwen3.5-VL releases.
_QWEN35_FALLBACK_TOKEN_IDS: dict[str, int] = {
	"image_token_id":        151655,
	"vision_start_token_id": 151652,
	"vision_end_token_id":   151653,
}

# token string → bare key name (arch prefix added in inject_kv)
_QWEN35_TOKEN_MAP: dict[str, str] = {
	"<|image_pad|>":    "image_token_id",
	"<|vision_start|>": "vision_start_token_id",
	"<|vision_end|>":   "vision_end_token_id",
}


# ---------------------------------------------------------------------------
# Module-level helper — token ID resolution via vocabulary scan
# ---------------------------------------------------------------------------

def _find_token_ids(llm_fields) -> dict[str, int]:
	"""
	Scan tokenizer.ggml.tokens in the text LLM GGUF to locate the Qwen3.5
	vision special-token IDs.  Falls back to known constants on failure so
	a corrupt or unusual tokenizer block does not hard-crash the merge.
	"""
	try:
		field = llm_fields["tokenizer.ggml.tokens"]
		tokens: list[str] = []
		for idx in field.data:
			raw = bytes(field.parts[idx])
			try:
				tokens.append(raw.decode("utf-8"))
			except UnicodeDecodeError:
				tokens.append(raw.decode("utf-8", errors="replace"))

		reverse: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}

		result: dict[str, int] = {}
		missing: list[str] = []
		for token_str, key in _QWEN35_TOKEN_MAP.items():
			if token_str in reverse:
				tid = reverse[token_str]
				result[key] = tid
				print(f"  [qwen35] token '{token_str}' → id {tid} → {key}")
			else:
				missing.append(token_str)

		if missing:
			raise KeyError(f"tokens not found in vocabulary: {missing}")

		return result

	except Exception as exc:  # noqa: BLE001
		print(f"  [qwen35] vocab scan failed ({exc}), using fallback token IDs")
		return dict(_QWEN35_FALLBACK_TOKEN_IDS)


def _deepstack_indexes_from_mmproj(mmproj_fields) -> list[int]:
	"""
	Derive vision.deepstack_visual_indexes from the mmproj bool array
	clip.vision.is_deepstack_layers.  Returns [] when absent (MoE / no
	deepstack).
	"""
	key = "clip.vision.is_deepstack_layers"
	if key not in mmproj_fields:
		return []
	flags = _read_array(mmproj_fields, key)
	return [i for i, v in enumerate(flags) if v]

def _build_kv_head_array(llm_fields, arch: str) -> list[int]:
	"""
	Reconstruct the per-layer attention.head_count_kv array from the
	text GGUF's scalar fields.  Mirrors convert_qwen3next.go kvHeadCounts():
	every layer at position (i+1) % full_attention_interval == 0 gets
	kv_heads; all others get 0.
	"""
	n        = int(_read_scalar(llm_fields, f"{arch}.block_count"))
	kv       = int(_read_scalar(llm_fields, f"{arch}.attention.head_count_kv"))
	interval = int(_read_scalar(llm_fields, f"{arch}.full_attention_interval"))
	return [kv if (i + 1) % interval == 0 else 0 for i in range(n)]


# ---------------------------------------------------------------------------
# Qwen3.5-VL Dense
# ---------------------------------------------------------------------------

class Qwen35ModelCore(QwenBaseModelCore):
	"""Merge plugin for Qwen3.5 dense models (e.g. 4B, 9B, 27B)."""

	MODEL_TYPE    = "qwen35"
	REQUIRES_BLOB = False
	STATUS        = "stable"

	@classmethod
	def get_help_info(cls) -> dict:  # pyright: ignore[reportMissingTypeArgument]
		return {
			"description":   "Qwen3.5 dense + vision hybrid models (e.g. 4B, 9B, 27B) — no blob required; also includes qwen3.6",
			"requires_blob": False,
			"status":        "stable",
			"extra_options": [],
		}

	# ------------------------------------------------------------------
	# KV Drop Set
	# ------------------------------------------------------------------

	def get_kv_drop(self) -> set[str]:
		# QwenBaseModelCore already drops: add_eos_token, add_padding_token,
		# eos_token_ids.  Drop keys that are re-injected below from llm_fields
		# or mmproj_fields so they are never duplicated in the output.
		a = self.arch
		return super().get_kv_drop() | {  # pyright: ignore[reportUnknownVariableType]
			f"{a}.attention.head_count_kv",   # re-injected from llm_fields
			f"{a}.image_token_id",             # re-injected via vocab scan
			f"{a}.vision_start_token_id",      # re-injected via vocab scan
			f"{a}.vision_end_token_id",        # re-injected via vocab scan
			f"{a}.vision.longest_edge",        # re-injected (hardcoded constant)
			f"{a}.vision.shortest_edge",       # re-injected (hardcoded constant)
			f"{a}.mrope_sections",             # re-injected from llm_fields
			f"{a}.rope.dimension_sections",    # re-injected from llm_fields
			f"{a}.rope.mrope_section",         # re-injected from llm_fields
			"tokenizer.ggml.padding_token_id", # re-injected below
			#"general.parameter_count",         # re-injected from llm_fields
		}

	# ------------------------------------------------------------------
	# KV Injection
	# ------------------------------------------------------------------

	def inject_kv(self, writer: GGUFWriter, ref_fields, mmproj_fields, llm_fields, *, args) -> None:
		self._llm_fields = llm_fields  # stash for get_llm_renames
		
		a = self.arch

		# -- Vision geometry (hardcoded — identical across all model sizes) --
		writer.add_uint32 (f"{a}.vision.num_channels",              3)
		writer.add_uint32 (f"{a}.vision.temporal_patch_size",       2)
		writer.add_uint32 (f"{a}.vision.num_positional_embeddings", 2304)
		writer.add_float32(f"{a}.vision.rope.freq_base",            10000.0)
		writer.add_uint32 (f"{a}.vision.longest_edge",  _VISION_LONGEST_EDGE)
		writer.add_uint32 (f"{a}.vision.shortest_edge", _VISION_SHORTEST_EDGE)

		# -- Token IDs (vocab scan → fallback to constants) --
		token_ids = _find_token_ids(llm_fields)
		writer.add_uint32(f"{a}.image_token_id",        token_ids["image_token_id"])
		writer.add_uint32(f"{a}.vision_start_token_id", token_ids["vision_start_token_id"])
		writer.add_uint32(f"{a}.vision_end_token_id",   token_ids["vision_end_token_id"])

		# -- RoPE / MRoPE
		# Text-only GGUFs only carry rope.dimension_sections (kv 20).
		# mrope_sections and rope.mrope_section are absent — use dimension_sections
		# as the canonical source and alias all three keys from it.
		dim_sec = _read_array(llm_fields, f"{a}.rope.dimension_sections")
		writer.add_array(f"{a}.rope.dimension_sections", dim_sec)
		writer.add_array(f"{a}.mrope_sections",          dim_sec)
		writer.add_array(f"{a}.rope.mrope_section",      dim_sec)
		writer.add_bool (f"{a}.rope.mrope_interleaved",  True)  # always True for qwen35-VL

		# -- SSM hybrid flag — always True for qwen35-VL; absent in text GGUF --
		writer.add_bool(f"{a}.ssm.v_head_reordered", True)

		# -- Attention head counts per layer
		# Text GGUF has a scalar u32; VL needs a per-layer array derived from
		# full_attention_interval and block_count.
		if f"{a}.attention.head_count_kv" in llm_fields:
			raw = llm_fields[f"{a}.attention.head_count_kv"]
			if hasattr(raw, '__len__') and len(raw.data) > 1:
				# Already a per-layer array (e.g. from a VL-converted GGUF)
				kv_heads = _read_array(llm_fields, f"{a}.attention.head_count_kv")
			else:
				# Scalar — rebuild per-layer array from full_attention_interval
				kv_heads = _build_kv_head_array(llm_fields, a)
		else:
			kv_heads = _build_kv_head_array(llm_fields, a)
		writer.add_array(f"{a}.attention.head_count_kv", kv_heads)

		# -- Tokenizer --
		writer.add_uint32("tokenizer.ggml.padding_token_id", 248044)
		writer.add_bool  ("tokenizer.ggml.add_eos_token",    False)
		writer.add_bool  ("tokenizer.ggml.add_padding_token", False)
		eos_ids = (
			_read_array(llm_fields, "tokenizer.ggml.eos_token_ids")
			if "tokenizer.ggml.eos_token_ids" in llm_fields
			else [int(_read_scalar(llm_fields, "tokenizer.ggml.eos_token_id"))]
		)
		writer.add_array("tokenizer.ggml.eos_token_ids", [int(x) for x in eos_ids])

		if "tokenizer.ggml.scores" in llm_fields:
			scores = _read_array(llm_fields, "tokenizer.ggml.scores")
			writer.add_array("tokenizer.ggml.scores", scores)

		# -- Deepstack indexes (mmproj bool array → int index list) --
		ds_indexes = _deepstack_indexes_from_mmproj(mmproj_fields)
		writer.add_array(f"{a}.vision.deepstack_visual_indexes",
						 [int(x) for x in ds_indexes])

		# -- General metadata --
		#writer.add_uint64("general.parameter_count",
		#                  int(_read_scalar(llm_fields, "general.parameter_count")))
		_ft = (
			int(_read_scalar(llm_fields, "general.file_type"))
			if "general.file_type" in llm_fields
			else 32
		)
		writer.add_uint32("general.file_type", _ft)

	# ------------------------------------------------------------------
	# LLM Tensor Renames — SSM dt bias suffix fix
	# ------------------------------------------------------------------

	def get_llm_renames(self, ref_fields=None, llm_fields=None) -> dict[str, str]:
		fields = llm_fields or ref_fields
		if fields is None:
			return {}
		a = self.arch
		# attention.head_count_kv in a text GGUF is a scalar u32, not a per-layer
		# array — use block_count directly to determine layer count instead.
		num_layers = int(_read_scalar(fields, f"{a}.block_count"))
		return {f"blk.{i}.ssm_dt.bias": f"blk.{i}.ssm_dt"
				for i in range(num_layers)}


# ---------------------------------------------------------------------------
# Qwen3.5-VL MoE
# ---------------------------------------------------------------------------

class Qwen35MoEModelCore(Qwen35ModelCore):
	"""
	Merge plugin for Qwen3.5-VL-MoE (e.g. 35B-A3B).

	Identical pipeline to Qwen35ModelCore.  MoE mmproj tensors carry no
	v.deepstack.* entries and clip.vision.is_deepstack_layers will be all
	False (or absent), so deepstack_indexes_from_mmproj returns [] and
	no deepstack renames are built in process_mmproj_tensors.
	"""

	MODEL_TYPE = "qwen35moe"
	STATUS     = "stable"

	@classmethod
	def get_help_info(cls) -> dict:  # pyright: ignore[reportMissingTypeArgument]
		return {
			"description":   "Qwen3.5-VL-MoE (e.g. 35B-A3B) — no blob required",
			"requires_blob": False,
			"status":        "stable",
			"extra_options": [],
		}
