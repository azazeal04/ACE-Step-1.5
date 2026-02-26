"""Sampling utilities for preprocessed tensor training datasets."""

from __future__ import annotations

import random
from typing import Dict, List


class BucketedBatchSampler:
    """Batch sampler that groups indices by latent-length buckets."""

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True) -> None:
        """Initialize bucket sampler.

        Args:
            lengths: Per-sample latent lengths.
            batch_size: Number of items per yielded batch.
            shuffle: Whether to shuffle buckets and samples each epoch.
        """
        self.lengths = lengths
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle

    def __iter__(self):
        """Yield batches of indices grouped by coarse latent-length buckets."""
        buckets: Dict[int, List[int]] = {}
        for idx, length in enumerate(self.lengths):
            bucket = int(length // 64)
            buckets.setdefault(bucket, []).append(idx)

        bucket_keys = list(buckets.keys())
        if self.shuffle:
            random.shuffle(bucket_keys)

        for key in bucket_keys:
            group = buckets[key]
            if self.shuffle:
                random.shuffle(group)
            for start in range(0, len(group), self.batch_size):
                yield group[start:start + self.batch_size]

    def __len__(self) -> int:
        """Return number of batches produced by bucketed iteration."""
        bucket_counts: Dict[int, int] = {}
        for length in self.lengths:
            bucket = int(length // 64)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        return sum((count + self.batch_size - 1) // self.batch_size for count in bucket_counts.values())
