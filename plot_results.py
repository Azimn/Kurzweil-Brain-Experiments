from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
summary_path = ROOT / "results" / "experiment_summary.json"
if not summary_path.exists():
    raise SystemExit("Run run_experiment.py first.")
summary = json.loads(summary_path.read_text(encoding="utf-8"))

conditions = list(summary["condition_summary"])
finals = [summary["condition_summary"][c]["mean_final_similarity"] for c in conditions]
stds = [summary["condition_summary"][c]["std_final_similarity"] for c in conditions]

fig, ax = plt.subplots(figsize=(8.5, 5.0))
ax.bar(np.arange(len(conditions)), finals, yerr=stds, capsize=4)
ax.set_xticks(np.arange(len(conditions)), conditions, rotation=18, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Adult phenotype similarity")
ax.set_title("Developmental Attractor: final held-out phenotype")
fig.tight_layout()
out = ROOT / "results" / "condition_similarity.png"
fig.savefig(out, dpi=180)
print(out)
