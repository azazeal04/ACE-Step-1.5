"""Tests for path-sanitisation in data_module.

Covers the safe_path integration in PreprocessedTensorDataset and
load_dataset_from_json that guards against path-traversal attacks
(CodeQL: uncontrolled data used in path expression).
"""

import os
import json
import random
import tempfile
import unittest
from unittest import mock

import torch

from acestep.training.path_safety import safe_path, set_safe_root
from acestep.training.data_module import (
    BucketedBatchSampler,
    PreprocessedTensorDataset,
    load_dataset_from_json,
)


class SafePathTests(unittest.TestCase):
    """Tests for safe_path from path_safety module."""

    def test_valid_directory(self):
        with tempfile.TemporaryDirectory() as d:
            parent = os.path.dirname(os.path.realpath(d))
            set_safe_root(parent)
            result = safe_path(d)
            self.assertEqual(result, os.path.realpath(d))

    def test_traversal_raises(self):
        with tempfile.TemporaryDirectory() as d:
            set_safe_root(d)
            with self.assertRaises(ValueError):
                safe_path("../../etc/passwd", base=d)

    def test_absolute_path_outside_raises(self):
        with tempfile.TemporaryDirectory() as d:
            set_safe_root(d)
            with self.assertRaises(ValueError):
                safe_path("/etc/passwd", base=d)

    def test_normal_child(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.realpath(d)
            result = safe_path("foo.pt", base=base)
            self.assertEqual(result, os.path.join(base, "foo.pt"))

    def test_absolute_path_inside_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.realpath(d)
            child = os.path.join(base, "sub", "file.pt")
            result = safe_path(child, base=base)
            self.assertEqual(result, child)


class PreprocessedTensorDatasetPathSafetyTests(unittest.TestCase):
    """Tests that PreprocessedTensorDataset rejects traversal paths."""

    def setUp(self):
        # Allow /tmp paths during tests
        set_safe_root(tempfile.gettempdir())

    def test_manifest_traversal_paths_skipped(self):
        """Paths in manifest.json that escape tensor_dir are ignored."""
        with tempfile.TemporaryDirectory() as d:
            # Create a manifest with one good and one bad path
            good_pt = os.path.join(d, "good.pt")
            open(good_pt, "wb").close()  # touch

            manifest = {
                "samples": [
                    "good.pt",
                    "../../etc/passwd",
                ]
            }
            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump(manifest, f)

            ds = PreprocessedTensorDataset(d)
            # Only the safe path should survive
            self.assertEqual(len(ds.valid_paths), 1)
            self.assertTrue(ds.valid_paths[0].endswith("good.pt"))

    def test_fallback_scan_only_finds_pt_files(self):
        """Without manifest, only .pt files inside tensor_dir are found."""
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.pt", "b.pt", "c.txt"]:
                open(os.path.join(d, name), "wb").close()

            ds = PreprocessedTensorDataset(d)
            self.assertEqual(len(ds.valid_paths), 2)

    def test_nonexistent_dir_raises(self):
        with self.assertRaises(ValueError):
            PreprocessedTensorDataset("/tmp/nonexistent_xyz_12345")

    def test_manifest_relative_to_tensor_dir(self):
        """Manifest with paths relative to tensor_dir loads correctly."""
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.pt", "b.pt"]:
                open(os.path.join(d, name), "wb").close()

            manifest = {"samples": ["a.pt", "b.pt"]}
            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump(manifest, f)

            ds = PreprocessedTensorDataset(d)
            self.assertEqual(len(ds.valid_paths), 2)

    def test_manifest_legacy_cwd_relative_paths(self):
        """Legacy manifest with CWD-relative paths resolves via fallback."""
        with tempfile.TemporaryDirectory() as root:
            set_safe_root(root)
            tensor_dir = os.path.join(root, "sub", "tensors")
            os.makedirs(tensor_dir)
            pt_file = os.path.join(tensor_dir, "sample.pt")
            open(pt_file, "wb").close()

            # Legacy manifest stored the full CWD-relative path
            legacy_rel = os.path.relpath(pt_file, root)
            manifest = {"samples": [legacy_rel]}
            with open(os.path.join(tensor_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f)

            ds = PreprocessedTensorDataset(tensor_dir)
            self.assertEqual(len(ds.valid_paths), 1)
            self.assertEqual(
                os.path.realpath(ds.valid_paths[0]),
                os.path.realpath(pt_file),
            )


class SaveManifestTests(unittest.TestCase):
    """Tests for save_manifest path normalisation."""

    def test_paths_stored_relative_to_output_dir(self):
        """save_manifest converts absolute/CWD-relative paths to dir-relative."""
        from acestep.training.dataset_builder_modules.preprocess_manifest import (
            save_manifest,
        )
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as d:
            metadata = SimpleNamespace(to_dict=lambda: {"name": "test"})
            # Simulate paths that preprocess_to_tensors produces
            output_paths = [
                os.path.join(d, "a.pt"),
                os.path.join(d, "b.pt"),
            ]
            save_manifest(d, metadata, output_paths)
            with open(os.path.join(d, "manifest.json")) as f:
                manifest = json.load(f)
            # Paths should be just filenames (relative to d)
            self.assertEqual(manifest["samples"], ["a.pt", "b.pt"])
            self.assertEqual(manifest["num_samples"], 2)

    def test_cwd_relative_input_normalised(self):
        """CWD-relative input paths are normalised to dir-relative."""
        from acestep.training.dataset_builder_modules.preprocess_manifest import (
            save_manifest,
        )
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as d:
            metadata = SimpleNamespace(to_dict=lambda: {"name": "test"})
            # Paths like "./subdir/a.pt" relative to CWD
            cwd_rel = os.path.relpath(os.path.join(d, "x.pt"))
            save_manifest(d, metadata, [cwd_rel])
            with open(os.path.join(d, "manifest.json")) as f:
                manifest = json.load(f)
            self.assertEqual(manifest["samples"], ["x.pt"])


class LoadDatasetFromJsonTests(unittest.TestCase):
    """Tests for load_dataset_from_json path validation."""

    def setUp(self):
        set_safe_root(tempfile.gettempdir())

    def test_nonexistent_file_raises(self):
        with self.assertRaises(ValueError):
            load_dataset_from_json("/tmp/nonexistent_file.json")

    def test_valid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"metadata": {"v": 1}, "samples": [{"a": 1}]}, f)
            path = f.name
        try:
            samples, meta = load_dataset_from_json(path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(meta["v"], 1)
        finally:
            os.unlink(path)


class AceStepDataModuleInitTests(unittest.TestCase):
    """Regression tests for legacy ``AceStepDataModule`` initialization."""

    def test_init_does_not_require_preprocessed_only_cache_args(self):
        """Legacy raw-audio datamodule should initialize without NameError."""
        from acestep.training.data_module import AceStepDataModule

        module = AceStepDataModule(samples=[], dit_handler=object())

        self.assertEqual(module.samples, [])
        self.assertIsNotNone(module.dit_handler)


class BucketedBatchSamplerTests(unittest.TestCase):
    """Tests for bucketed batching semantics and sizing."""

    def test_len_counts_batches_per_bucket(self):
        """Length should be the sum of per-bucket ceiling batch counts."""
        lengths = [10, 20, 30, 65, 130]
        sampler = BucketedBatchSampler(lengths=lengths, batch_size=2, shuffle=False)

        self.assertEqual(len(sampler), 4)

    def test_shuffle_changes_order_but_preserves_batch_sizes(self):
        """Shuffling should only affect order, not total sample coverage."""
        lengths = [8, 9, 70, 72, 140, 141]

        baseline = list(BucketedBatchSampler(lengths=lengths, batch_size=2, shuffle=False))

        random.seed(1234)
        shuffled = list(BucketedBatchSampler(lengths=lengths, batch_size=2, shuffle=True))

        baseline_indices = sorted(idx for batch in baseline for idx in batch)
        shuffled_indices = sorted(idx for batch in shuffled for idx in batch)

        self.assertEqual(baseline_indices, shuffled_indices)
        self.assertTrue(all(0 < len(batch) <= 2 for batch in shuffled))


class PreprocessedTensorDatasetCachingTests(unittest.TestCase):
    """Tests for preprocessed dataset LRU cache behavior and latent lengths."""

    def setUp(self):
        set_safe_root(tempfile.gettempdir())

    def _write_sample(self, path: str, length: int) -> None:
        """Write a minimal valid tensor sample for test dataset usage."""
        sample = {
            "target_latents": torch.zeros(length, 64),
            "attention_mask": torch.ones(length),
            "encoder_hidden_states": torch.zeros(2, 4),
            "encoder_attention_mask": torch.ones(2),
            "context_latents": torch.zeros(length, 65),
        }
        torch.save(sample, path)

    def test_latent_lengths_from_valid_and_invalid_files(self):
        """Dataset should record true lengths and use 0 for invalid tensors."""
        with tempfile.TemporaryDirectory() as d:
            valid = os.path.join(d, "valid.pt")
            invalid = os.path.join(d, "invalid.pt")
            self._write_sample(valid, 7)
            torch.save({"not_target_latents": torch.ones(1)}, invalid)

            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump({"samples": ["valid.pt", "invalid.pt"]}, f)

            ds = PreprocessedTensorDataset(d)
            self.assertEqual(ds.latent_lengths, [7, 0])

    def test_ram_lru_cache_eviction(self):
        """LRU cache should evict the least-recently used item at capacity."""
        with tempfile.TemporaryDirectory() as d:
            for name, length in (("a.pt", 5), ("b.pt", 6), ("c.pt", 7)):
                self._write_sample(os.path.join(d, name), length)

            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump({"samples": ["a.pt", "b.pt", "c.pt"]}, f)

            ds = PreprocessedTensorDataset(d, cache_policy="ram_lru", cache_max_items=2)
            with mock.patch("acestep.training.data_module.torch.load", wraps=torch.load) as load_mock:
                _ = ds[0]
                _ = ds[1]
                _ = ds[0]
                _ = ds[2]
                _ = ds[1]

            self.assertEqual(load_mock.call_count, 4)
            self.assertEqual(list(ds._cache.keys()), [2, 1])



if __name__ == "__main__":
    unittest.main()
