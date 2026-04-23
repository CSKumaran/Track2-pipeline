# Pipeline v2.2 — Temporal Contiguity Analysis for Instructional Videos
__version__ = "2.2.0"

# === PyTorch 2.6 global compat fix ===
# Force weights_only=False ALWAYS — speechbrain/whisperx pass it explicitly
import torch as _torch
_orig_load = _torch.load
def _safe_load(*args, **kwargs):
    kwargs["weights_only"] = False  # override even if explicitly set True
    return _orig_load(*args, **kwargs)
_torch.load = _safe_load
