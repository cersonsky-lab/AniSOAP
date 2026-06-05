import torch


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions [w, x, y, z] to rotation matrices.

    Parameters
    ----------
    q
        Tensor with shape (..., 4), using scalar-first convention.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape (..., 3, 3).
    """
    q = torch.as_tensor(q)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )
