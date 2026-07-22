import torch
import torch.distributions as dist


def sample_synthetic_extreme_data(n, alpha=1.5, beta=0.6, device="cpu"):
    """
    Generate synthetic data in dimension 5.

    R = 2 U * InvGamma(alpha, beta)
    W | R=r ~ Dirichlet(a1(r), a1(r), a2(r), a2(r), a2(r))
    X = R W
    """

    U = torch.rand(n, 1, device=device)

    # If Y ~ Gamma(alpha, beta), then 1/Y ~ InvGamma(alpha, beta)
    gamma = dist.Gamma(
        concentration=torch.tensor(alpha, device=device),
        rate=torch.tensor(beta, device=device)
    )

    inv_gamma = 1.0 / gamma.sample((n,)).view(n, 1)

    R = 2.0 * U * inv_gamma

    m = torch.minimum(
        torch.ones_like(R),
        1.0 / (2.0 * R)
    )

    a1 = 3.0 * (2.0 - m)
    a2 = 3.0 * (1.0 + m)

    concentration = torch.cat(
        [a1, a1, a2, a2, a2],
        dim=1
    )

    W = dist.Dirichlet(concentration).sample()

    X = R * W

    return X, R, W