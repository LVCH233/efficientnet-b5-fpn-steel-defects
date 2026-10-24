"""Iterative multi-label splits at the duplicate-content group level."""
from collections import defaultdict
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def create_splits(records, seed=20261008):
    groups = defaultdict(list)
    seen = set()
    for row in records:
        if row['image_id'] in seen:
            raise ValueError(f'Duplicate image identifier: {row["image_id"]}')
        seen.add(row['image_id'])
        groups[row['sha256']].append(row)
    keys = sorted(groups)
    labels = np.array([[max(int(row[f'c{c}']) for row in groups[key])
                        for c in range(1, 5)] for key in keys])
    indices = np.arange(len(keys))
    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train, held = next(first.split(indices[:, None], labels))
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed+1)
    val_relative, test_relative = next(second.split(held[:, None], labels[held]))
    assignments = {}
    for name, selected in [('train', train), ('val', held[val_relative]), ('test', held[test_relative])]:
        for index in selected:
            for row in groups[keys[index]]:
                assignments[row['image_id']] = name
    if set(assignments) != seen:
        raise ValueError('Split lost images')
    return assignments
