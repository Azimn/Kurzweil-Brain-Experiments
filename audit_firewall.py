from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data" / "development"
EVAL = ROOT / "evaluation" / "adult_phenotype_probes.json"
FORBIDDEN_KEYS = {"target_action_weights", "expected_tendencies", "adult_phenotype", "historical_action"}


def walk(obj, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                raise AssertionError(f"Forbidden evaluation/training-leak key {k!r} at {path}")
            walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]")


for f in sorted(DEV.glob("*.json")):
    payload = json.loads(f.read_text(encoding="utf-8"))
    walk(payload, f.name)
    for event in payload.get("events", []):
        assert "situation" in event
        assert "consequences" in event
        assert "action" not in event, f"Historical/supervised action field leaked into {f.name}:{event.get('id')}"

blob = EVAL.read_bytes()
print("Developmental firewall: PASS")
print(f"Development files checked: {len(list(DEV.glob('*.json')))}")
print(f"Evaluation answer-key SHA256: {hashlib.sha256(blob).hexdigest()}")
