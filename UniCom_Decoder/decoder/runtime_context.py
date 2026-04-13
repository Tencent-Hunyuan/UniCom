from typing import Any


_RUNTIME_ARGS: Any = None


def set_runtime_args(args) -> None:
    global _RUNTIME_ARGS
    _RUNTIME_ARGS = args


def get_runtime_args():
    if _RUNTIME_ARGS is None:
        raise RuntimeError("Runtime args have not been initialized.")
    return _RUNTIME_ARGS


def print_rank_0(*args, **kwargs):
    print(*args, **kwargs)
