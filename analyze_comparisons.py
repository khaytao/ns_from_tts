import json
import numpy as np
from collections import Counter
from typing import Dict, List, Any


def analyze_results(json_path: str) -> Dict[str, Any]:
    """
    Load the JSON results, flatten them, compute statistics, and print diagnostics.
    Returns a dictionary with all computed results.
    """
    # ---- Load JSON ----
    with open(json_path, "r") as f:
        data = json.load(f)

    # ---- Flatten nested structure ----
    flat = [item for sublist in data for item in sublist]

    # ---- Show unique "closer" values ----
    unique_closer = set(d["closer"] for d in flat)
    print("Unique 'closer' values:", unique_closer)

    # ---- Find all entries labeled 'base' ----
    base_entries = []
    for i, d in enumerate(flat):
        if d["closer"] == "base":
            base_entries.append((i, d))

    if base_entries:
        print("\nEntries labeled 'base':")
        for idx, entry in base_entries:
            print(f"Index {idx}: {entry}")
    else:
        print("\nNo 'base' entries found.")

    # ---- Extract distances ----
    dist_control = [d["distance_with control"] for d in flat]
    dist_base = [d["distance_base"] for d in flat]
    closer_list = [d["closer"] for d in flat]

    # ---- Compute statistics ----
    mean_control = float(np.mean(dist_control))
    std_control = float(np.std(dist_control))

    mean_base = float(np.mean(dist_base))
    std_base = float(np.std(dist_base))

    closer_counts = Counter(closer_list)

    # ---- Print results ----
    print("\n=== Distance With Control ===")
    print(f"Mean: {mean_control:.4f}")
    print(f"Std:  {std_control:.4f}")

    print("\n=== Distance Base ===")
    print(f"Mean: {mean_base:.4f}")
    print(f"Std:  {std_base:.4f}")

    print("\n=== Closer Counts ===")
    for k, v in closer_counts.items():
        print(f"{k}: {v}")

    # ---- Return analysis ----
    return {
        "unique_closer": unique_closer,
        "base_entries": base_entries,
        "stats": {
            "control": {"mean": mean_control, "std": std_control},
            "base": {"mean": mean_base, "std": std_base},
        },
        "closer_counts": dict(closer_counts),
    }
