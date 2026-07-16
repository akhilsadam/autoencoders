import torch


def vector_rms_norm(z, zero_mean=False, eps=1e-6):
    """RMS normalization: scale by inverse square root of mean square."""
    assert z.ndim in [3, 4]  # [B, C, H, W] or [B, N, D]
    dim = tuple(range(1, z.ndim))  # all dims except batch
    if zero_mean:
        z = z - z.mean(dim=dim, keepdim=True)
    m = z.square().mean(dim=dim, keepdim=True)
    m = torch.rsqrt(m + eps).to(z.dtype)
    return z * m


def std_norm(z, zero_mean=True, eps=1e-6):
    """Standard deviation normalization: zero mean and unit variance."""
    assert z.ndim in [3, 4]  # [B, C, H, W] or [B, N, D]
    dim = tuple(range(1, z.ndim))  # all dims except batch
    if zero_mean:
        z = z - z.mean(dim=dim, keepdim=True)
    z = z / (z.std(dim=dim, keepdim=True) + eps)
    return z
