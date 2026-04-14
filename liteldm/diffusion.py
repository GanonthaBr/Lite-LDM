import torch


class DDPMScheduler:
    """Linear beta schedule with forward and reverse DDPM steps."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = self.alpha_bar.sqrt()
        self.sqrt_one_m_ab = (1 - self.alpha_bar).sqrt()

    def add_noise(self, z0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(z0)
        ab = self.sqrt_ab[t][:, None, None, None]
        s1mab = self.sqrt_one_m_ab[t][:, None, None, None]
        return ab * z0 + s1mab * eps, eps

    @torch.no_grad()
    def sample_step(self, model, z_t: torch.Tensor, t_scalar: int) -> torch.Tensor:
        batch_size = z_t.shape[0]
        t_vec = torch.full((batch_size,), t_scalar, device=z_t.device, dtype=torch.long)

        eps_pred = model(z_t, t_vec)

        beta = self.betas[t_scalar]
        alpha = self.alphas[t_scalar]
        ab = self.alpha_bar[t_scalar]

        coef1 = 1.0 / alpha.sqrt()
        coef2 = beta / (1 - ab).sqrt()
        z_prev = coef1 * (z_t - coef2 * eps_pred)

        if t_scalar > 0:
            z_prev = z_prev + beta.sqrt() * torch.randn_like(z_t)
        return z_prev
