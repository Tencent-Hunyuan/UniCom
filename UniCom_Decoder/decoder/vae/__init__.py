from pathlib import Path

from .flux_vae import FluxAutoencoderKL
from ..constants import VAE_PATH


def load_vae(vae_type, vae_precision=None, sample_size=None, vae_path=None, logger=None, device=None):
    del vae_precision, sample_size
    if vae_type != "88-16c-flux":
        raise ValueError(f"Only Flux VAE is supported in this decoder runtime, got {vae_type}")

    if vae_path is None:
        vae_path = VAE_PATH[vae_type]
    if logger is not None:
        logger.info(f"Loading Flux VAE from: {vae_path}")

    vae = FluxAutoencoderKL.from_pretrained(Path(vae_path))
    vae.requires_grad_(False)
    if device is not None:
        vae = vae.to(device)
    vae.eval()
    return vae, vae_path, 8, 1
