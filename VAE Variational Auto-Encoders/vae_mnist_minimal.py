"""Minimal VAE (Variational Auto-Encoder, VAE) on MNIST using PyTorch.

Run (Windows / PowerShell):
  pip install -r requirements.txt
  python vae_mnist_minimal.py --epochs 5

Outputs:
    - out/samples_epoch_*.png  (generated samples)
    - out/recon_epoch_*.png    (top: originals, bottom: reconstructions)
    - out/vae_mnist.pt         (model weights)

This is intentionally minimal: MLP encoder/decoder + ELBO (BCE + KL).

Important path note:
    The output folder (out/) and dataset cache (data/) are created under the
    script directory (VAE Variational Auto-Encoders/) by default. This avoids
    accidentally creating out/ under your current working directory.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    epochs: int
    latent_dim: int
    lr: float
    seed: int
    log_interval: int
    num_workers: int
    out_dir: str


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc_fc1 = nn.Linear(28 * 28, 400)
        self.enc_mu = nn.Linear(400, latent_dim)
        self.enc_logvar = nn.Linear(400, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, 400)
        self.dec_fc2 = nn.Linear(400, 28 * 28)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder: x -> (mu, logvar).

        In a vanilla Auto-Encoder (AE), the encoder outputs a single latent vector z.
        In VAE, the encoder outputs parameters of a distribution q(z|x), typically a
        diagonal Gaussian N(mu, diag(sigma^2)).

        We output log-variance (logvar = log(sigma^2)) for numerical stability.
        """
        h = F.relu(self.enc_fc1(x))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization Trick (RT): sample z ~ N(mu, sigma^2) in a differentiable way.

        Direct sampling z ~ q_phi(z|x) would break backpropagation because sampling is
        not a deterministic function of the network parameters.

        We rewrite sampling as:
            eps ~ N(0, I)
            z = mu + sigma * eps, where sigma = exp(0.5 * logvar)

        Now the randomness lives in eps (parameter-free), while mu/logvar remain fully
        differentiable outputs of the encoder.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder: z -> x_hat.

        For MNIST, pixels are in [0, 1]. A common choice is a Bernoulli likelihood
        p(x|z) with parameter x_hat in (0, 1), so we apply sigmoid.
        """
        h = F.relu(self.dec_fc1(z))
        x_logits = self.dec_fc2(h)
        # For MNIST in [0,1], use sigmoid to get Bernoulli probabilities.
        return torch.sigmoid(x_logits)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass used during training.

        Returns:
            recon: reconstructed x_hat
            mu, logvar: parameters for q(z|x)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def elbo_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute negative ELBO (Evidence Lower Bound, ELBO) as a loss.

    ELBO for one sample x is:
      E_{q(z|x)}[ log p(x|z) ] - KL(q(z|x) || p(z))

    Training usually MINIMIZES the negative ELBO:
      loss = recon_loss + kl_loss

    - Reconstruction term:
        We model p(x|z) as Bernoulli with probabilities recon_x in (0,1).
        Then -log p(x|z) becomes Binary Cross Entropy (BCE).
    - KL term:
        q(z|x) is diagonal Gaussian N(mu, diag(sigma^2))
        p(z) is standard normal N(0, I)
        KL has a closed-form expression.

    Returns (total, recon_bce, kl) summed over the batch.
    """

    # Reconstruction loss (sum over all pixels and batch)
    recon_bce = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL(q(z|x) || p(z)) where p(z)=N(0, I) and q is diagonal Gaussian.
    # Closed-form: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0)

    total = recon_bce + kl
    return total, recon_bce, kl


@torch.no_grad()
def save_reconstructions(model: VAE, device: torch.device, loader: DataLoader, out_path: str) -> None:
    """Save a visualization grid: originals (top row group) + reconstructions (bottom group)."""
    model.eval()
    batch = next(iter(loader))
    x, _ = batch
    x = x.to(device)
    x_flat = x.view(x.size(0), -1)

    recon, _, _ = model(x_flat)
    recon_img = recon.view(-1, 1, 28, 28)

    # stack originals (top) + recon (bottom)
    grid = torch.cat([x[:16], recon_img[:16]], dim=0)
    save_image(grid.cpu(), out_path, nrow=16)


@torch.no_grad()
def save_samples(model: VAE, device: torch.device, latent_dim: int, out_path: str, n: int = 64) -> None:
    """Sample z ~ N(0, I) and decode to images, then save a grid."""
    model.eval()
    z = torch.randn(n, latent_dim, device=device)
    samples = model.decode(z).view(-1, 1, 28, 28)
    save_image(samples.cpu(), out_path, nrow=8)


def parse_args() -> TrainConfig:
    """Parse CLI arguments.

    Note: --out-dir is treated as relative to this script directory unless it is an
    absolute path.
    """
    p = argparse.ArgumentParser(description="Minimal VAE on MNIST")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=100)
    # On Windows, num_workers=0 avoids common DataLoader multiprocessing issues.
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="out")

    a = p.parse_args()
    return TrainConfig(
        batch_size=a.batch_size,
        epochs=a.epochs,
        latent_dim=a.latent_dim,
        lr=a.lr,
        seed=a.seed,
        log_interval=a.log_interval,
        num_workers=a.num_workers,
        out_dir=a.out_dir,
    )


def main() -> None:
    cfg = parse_args()

    # Make paths robust: anchor relative paths to the script directory.
    # This ensures out/ is created under "VAE Variational Auto-Encoders/" even if you
    # run the script from the repo root or any other directory.
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(cfg.out_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset cache directory (downloaded MNIST files)
    data_dir = script_dir / "data"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = VAE(latent_dim=cfg.latent_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Initial visualization before training (optional but helpful)
    save_samples(model, device, cfg.latent_dim, str(out_dir / "samples_epoch_0.png"))
    save_reconstructions(model, device, test_loader, str(out_dir / "recon_epoch_0.png"))

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for batch_idx, (x, _) in enumerate(train_loader, start=1):
            # x: [B, 1, 28, 28] in [0, 1]
            x = x.to(device)

            # Flatten for the MLP encoder/decoder: [B, 784]
            x_flat = x.view(x.size(0), -1)

            recon, mu, logvar = model(x_flat)
            loss, recon_bce, kl = elbo_loss(recon, x_flat, mu, logvar)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_recon += recon_bce.item()
            total_kl += kl.item()

            if batch_idx % cfg.log_interval == 0:
                # Here we print a per-sample average loss estimate for readability.
                # (We used reduction="sum" above, so divide back by number of samples.)
                avg = total_loss / (batch_idx * cfg.batch_size)
                print(f"Epoch {epoch:02d} [{batch_idx:04d}/{len(train_loader):04d}]  loss={avg:.4f} (per-sample)")

        # End-of-epoch: evaluation visuals
        save_samples(model, device, cfg.latent_dim, str(out_dir / f"samples_epoch_{epoch}.png"))
        save_reconstructions(model, device, test_loader, str(out_dir / f"recon_epoch_{epoch}.png"))

        n_train = len(train_ds)
        print(
            f"Epoch {epoch:02d} done | "
            f"loss={total_loss / n_train:.4f} "
            f"recon={total_recon / n_train:.4f} "
            f"kl={total_kl / n_train:.4f} (per-sample)"
        )

    torch.save(model.state_dict(), str(out_dir / "vae_mnist.pt"))
    print(f"Saved model to: {out_dir / 'vae_mnist.pt'}")


if __name__ == "__main__":
    main()
