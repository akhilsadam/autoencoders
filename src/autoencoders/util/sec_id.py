import datetime
from omegaconf import OmegaConf

def _compact_sec_id() -> str:
    """Generate compact base36 sec ID (unique, sortable, 6 chars)."""
    dt = datetime.datetime.now()
    epoch = datetime.datetime(2025, 1, 1)
    delta = int((dt - epoch).total_seconds())
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    s = ""
    while delta:
        delta, i = divmod(delta, 36)
        s = chars[i] + s
    return s or "0"

# Register for Hydra
OmegaConf.register_new_resolver("sec_id", lambda: _compact_sec_id())
