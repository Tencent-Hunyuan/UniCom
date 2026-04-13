from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from UniCom_Decoder.runtime import run_cli
else:
    from UniCom_Decoder.runtime import run_cli


if __name__ == "__main__":
    run_cli()
