"""Model deployment script — package fine-tuned model and wire into inference service."""

import shutil
from pathlib import Path


def deploy(model_source: Path, target: Path = Path("models/cb-sentiment-v1/final")) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in model_source.glob("*"):
        dest = target / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print(f"Model deployed to {target}")


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("training/output")
    deploy(src)
