"""Model version registry — register and list deployed models."""

import json
from datetime import datetime, timezone
from pathlib import Path


METADATA_PATH = Path("models/metadata.json")


def register(
    model_name: str,
    version: str,
    metrics: dict,
    training_date: str,
    dataset_hash: str,
    git_sha: str,
) -> None:
    data = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {"version": "0.1.0", "models": []}
    entry = {
        "model_name": model_name,
        "version": version,
        "training_date": training_date,
        "dataset_hash": dataset_hash,
        "metrics": metrics,
        "git_sha": git_sha,
        "deployment_date": datetime.now(timezone.utc).isoformat(),
    }
    data["models"].append(entry)
    METADATA_PATH.write_text(json.dumps(data, indent=2))


def list_models() -> list[dict]:
    if not METADATA_PATH.exists():
        return []
    return json.loads(METADATA_PATH.read_text()).get("models", [])


if __name__ == "__main__":
    print("Models deployed:")
    for m in list_models():
        print(f"  {m.get('model_name', '?')} v{m.get('version', '?')} — "
              f"deployed {m.get('deployment_date', '?')[:16]}")
