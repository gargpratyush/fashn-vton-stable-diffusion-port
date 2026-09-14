"""Validate dynamic study selection before generating labels or selecting result rows."""
LABELS = {"bf16": "BF16 storage / F32 compute", "q8_0": "Q8_0", "q4_0": "Q4_0",
          "q5_0": "Q5_0", "q4_K": "Q4_K", "q5_K": "Q5_K"}


def policy_labels(state):
    selected = state["selection"]
    if selected not in ("q4_0", "q5_0", "q4_K", "q5_K"):
        raise ValueError("Unknown selected lower-bit policy")
    mixed = state["weights"]["selected-mixed"]
    restored = mixed["policy"]["f32_matrices"]
    if mixed["matrix_type"] != selected or not restored or len(set(restored)) != len(restored):
        raise ValueError("Mixed policy disagrees with recorded selection")
    labels = dict(LABELS)
    labels["selected-mixed"] = f"{LABELS[selected]} + {len(restored)} original F32 matrices"
    required = [f"full-cardigan-{selected}-s42", f"full-cardigan-{selected}-s43",
                f"repeat-cardigan-{selected}-1", f"repeat-cardigan-{selected}-2"]
    if any(name not in state["jobs"] for name in required):
        raise ValueError("Selected-policy repeat/seed rows are missing")
    return labels
