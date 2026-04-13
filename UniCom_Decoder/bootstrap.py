import importlib
import sys
from pathlib import Path


DECODER_SOURCE_ROOT = Path(__file__).resolve().parent / "decoder"


def bootstrap(project_root: Path, args):
    del project_root
    source_root = DECODER_SOURCE_ROOT.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Embedded decoder runtime not found: {source_root}")

    package_root = source_root.parent

    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    runtime_context_mod = importlib.import_module("decoder.runtime_context")
    runtime_context_mod.set_runtime_args(args)

    importlib.import_module("decoder.models.modules")
    model_mod = importlib.import_module("decoder.models.model")
    posemb_mod = importlib.import_module("decoder.models.modules.posemb_layers")
    return source_root, model_mod, posemb_mod
