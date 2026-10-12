"""Use Ascend's existing per-group mapping without cached raw address tables."""
from functools import wraps


def install_per_group_slot_mapping(block_table_cls=None):
    if block_table_cls is None:
        from vllm_ascend.worker.block_table import MultiGroupBlockTable
        block_table_cls = MultiGroupBlockTable
    original = block_table_cls.compute_slot_mapping
    if getattr(original, '_v41_per_group', False):
        return

    @wraps(original)
    def compute_slot_mapping(self, *args, **kwargs):
        # Disable on every call, including after sleep/wake. The pinned method
        # already implements the V4-style per-group and compressed fallbacks.
        self._can_fuse_slot_mapping = False
        return original(self, *args, **kwargs)

    compute_slot_mapping._v41_per_group = True
    block_table_cls.compute_slot_mapping = compute_slot_mapping
