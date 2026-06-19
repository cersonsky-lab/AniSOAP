import torch

from anisoap.representations.ellipsoidal_density_projection import (
    compute_moments,
    compute_moments_batched,
)


def test_compute_moments_batched_matches_single():
    dtype = torch.float64

    A = torch.tensor(
        [
            [[2.0, 0.1, 0.0], [0.1, 1.7, 0.2], [0.0, 0.2, 1.4]],
            [[1.5, -0.05, 0.1], [-0.05, 2.2, 0.0], [0.1, 0.0, 1.8]],
        ],
        dtype=dtype,
    )
    centers = torch.tensor(
        [
            [0.2, -0.1, 0.4],
            [-0.3, 0.5, 0.1],
        ],
        dtype=dtype,
    )
    maxdeg = 5

    batched, exponents = compute_moments_batched(A, centers, maxdeg)

    for i in range(A.shape[0]):
        single_cube = compute_moments(A[i], centers[i], maxdeg)
        expected = torch.stack(
            [
                single_cube[int(px), int(py), int(pz)]
                for px, py, pz in exponents.detach().cpu().tolist()
            ]
        ).to(dtype=dtype)

        torch.testing.assert_close(
            batched[i],
            expected,
            rtol=1e-10,
            atol=1e-10,
        )


def test_compute_moments_batched_has_gradients():
    dtype = torch.float64

    raw = torch.tensor(
        [
            [[1.5, 0.1, 0.0], [0.1, 1.4, 0.2], [0.0, 0.2, 1.8]],
            [[1.7, -0.1, 0.1], [-0.1, 1.9, 0.0], [0.1, 0.0, 1.6]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    A = raw @ raw.transpose(-1, -2) + 0.5 * torch.eye(3, dtype=dtype)

    centers = torch.tensor(
        [
            [0.2, -0.1, 0.4],
            [-0.3, 0.5, 0.1],
        ],
        dtype=dtype,
        requires_grad=True,
    )

    moments, _ = compute_moments_batched(A, centers, maxdeg=5)
    loss = moments.square().sum()
    loss.backward()

    assert raw.grad is not None
    assert centers.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert torch.isfinite(centers.grad).all()
    assert raw.grad.abs().sum() > 0
    assert centers.grad.abs().sum() > 0
