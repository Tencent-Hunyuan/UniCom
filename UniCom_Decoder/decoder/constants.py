from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ZOO = PROJECT_ROOT / "model_zoo"

VAE_PATH = {
    "88-16c-flux": str(DEFAULT_MODEL_ZOO / "flux-vae"),
}

__all__ = ["VAE_PATH"]
