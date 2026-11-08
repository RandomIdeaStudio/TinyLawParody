"""TinyLawParody: a tiny GPT trained from scratch that writes parody Czech laws."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
WEB_MODEL_DIR = ROOT / "web" / "model"
EOT = "<|endoftext|>"


def version() -> str:
    """Read the project version from pyproject.toml (the only place it lives)."""
    import tomllib

    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]
