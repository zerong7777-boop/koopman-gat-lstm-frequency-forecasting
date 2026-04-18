from __future__ import annotations

import math
import random


def split_case_ids(case_ids, train, val, test, seed) -> dict[str, list[str]]:
    case_ids = list(case_ids)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case_ids must be unique")
    if min(train, val, test) < 0:
        raise ValueError("split fractions must be non-negative")
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    shuffled = case_ids.copy()
    random.Random(seed).shuffle(shuffled)

    fractions = [train, val, test]
    exact_counts = [len(shuffled) * fraction for fraction in fractions]
    counts = [math.floor(count) for count in exact_counts]
    remainder = len(shuffled) - sum(counts)
    if remainder < 0:
        raise ValueError("split fractions produce invalid counts")

    remainder_order = sorted(
        (index for index, fraction in enumerate(fractions) if fraction > 0),
        key=lambda index: (exact_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in remainder_order[:remainder]:
        counts[index] += 1

    train_count, val_count, _ = counts

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }
