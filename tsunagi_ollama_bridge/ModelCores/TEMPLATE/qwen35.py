"""
ModelCores/qwen35.py
====================
ModelCore plugin stub for Qwen3.5-VL (dense + SSM hybrid, e.g. 4B/7B)
and Qwen3.5-VL-MoE (e.g. 35B-A3B).

Status: STUB — not yet migrated from the monolith.

Port checklist (see monolith for reference implementations)
-----------------------------------------------------------
[ ] MODEL_TYPE = "qwen35" / "qwen35moe"
[ ] REQUIRES_BLOB = True
[ ] get_kv_drop()          extend with qwen35-specific keys (mrope, edges, etc.)
[ ] get_kv_renames()       standard clip.vision.* map
[ ] inject_kv()            port inject_kv_qwen35() — most values from blob
[ ] get_llm_renames()      ssm_dt.bias → ssm_dt renames (num_layers from blob)
[ ] process_mmproj_tensors() QKV split + deepstack + patch_embed stack
                             (shared Qwen logic — consider a QwenMixinCore)
"""

from .base import BaseModelCore


class Qwen35ModelCore(BaseModelCore):
    MODEL_TYPE    = "qwen35"
    REQUIRES_BLOB = True
    STATUS        = "stub"

    @classmethod
    def get_help_info(cls) -> dict:
        return {
            "description":   "Qwen3.5-VL dense + SSM hybrid (e.g. 4B, 7B)",
            "requires_blob": True,
            "status":        "stub",
            "extra_options": [],
        }

    def inject_kv(self, writer, ref_fields, mmproj_fields, llm_fields, *, args):
        raise NotImplementedError(
            "Qwen3.5-VL has not yet been ported to the plugin system. "
            "Use the original monolith script for now."
        )

    def process_mmproj_tensors(self, mmproj, args) -> dict:
        raise NotImplementedError(
            "Qwen3.5-VL has not yet been ported to the plugin system. "
            "Use the original monolith script for now."
        )


class Qwen35MoEModelCore(BaseModelCore):
    MODEL_TYPE    = "qwen35moe"
    REQUIRES_BLOB = True
    STATUS        = "stub"

    @classmethod
    def get_help_info(cls) -> dict:
        return {
            "description":   "Qwen3.5-VL-MoE (e.g. 35B-A3B)",
            "requires_blob": True,
            "status":        "stub",
            "extra_options": [],
        }

    def inject_kv(self, writer, ref_fields, mmproj_fields, llm_fields, *, args):
        raise NotImplementedError(
            "Qwen3.5-VL-MoE has not yet been ported to the plugin system. "
            "Use the original monolith script for now."
        )

    def process_mmproj_tensors(self, mmproj, args) -> dict:
        raise NotImplementedError(
            "Qwen3.5-VL-MoE has not yet been ported to the plugin system. "
            "Use the original monolith script for now."
        )
