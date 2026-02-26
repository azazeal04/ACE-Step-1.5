"""Unit tests for ``trainer_helpers.resume_checkpoint`` full-mode fallback."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from acestep.training_v2 import trainer_helpers


class ResumeCheckpointFullModeTests(unittest.TestCase):
    """Validate full-mode resume behavior when full checkpoint file is absent."""

    def test_full_mode_missing_decoder_state_yields_warning(self) -> None:
        """Missing ``full_decoder_state.pt`` should warn and return ``None``."""
        trainer = SimpleNamespace(
            module=SimpleNamespace(device=torch.device("cpu"), lycoris_net=None),
            training_config=SimpleNamespace(training_mode="full"),
            adapter_type="lora",
        )
        optimizer = SimpleNamespace(load_state_dict=lambda _state: None)
        scheduler = SimpleNamespace(load_state_dict=lambda _state: None)

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = trainer_helpers.resume_checkpoint(
                trainer,
                tmpdir,
                optimizer,
                scheduler,
            )
            updates = []
            try:
                while True:
                    updates.append(next(generator))
            except StopIteration as stop:
                result = stop.value

        self.assertIsNone(result)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].kind, "warn")
        self.assertIn("full_decoder_state.pt not found", updates[0].msg)

    def test_full_mode_resume_restores_training_state(self) -> None:
        """Full-mode resume should restore epoch/step and optimizer/scheduler states."""
        model = torch.nn.Module()
        model.decoder = torch.nn.Linear(4, 4)
        trainer = SimpleNamespace(
            module=SimpleNamespace(
                device=torch.device("cpu"),
                lycoris_net=None,
                model=model,
            ),
            training_config=SimpleNamespace(training_mode="full"),
            adapter_type="lora",
        )
        optimizer = SimpleNamespace(load_state_dict=mock.Mock())
        scheduler = SimpleNamespace(load_state_dict=mock.Mock())
        optimizer_state = {"param_groups": [{"lr": 1e-4}]}
        scheduler_state = {"last_epoch": 9}

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model.decoder.state_dict(), f"{tmpdir}/full_decoder_state.pt")
            torch.save(
                {
                    "epoch": 3,
                    "global_step": 123,
                    "optimizer_state_dict": optimizer_state,
                    "scheduler_state_dict": scheduler_state,
                },
                f"{tmpdir}/training_state.pt",
            )

            generator = trainer_helpers.resume_checkpoint(
                trainer,
                tmpdir,
                optimizer,
                scheduler,
            )
            updates = []
            try:
                while True:
                    updates.append(next(generator))
            except StopIteration as stop:
                result = stop.value

        self.assertEqual(result, (3, 123))
        optimizer.load_state_dict.assert_called_once_with(optimizer_state)
        scheduler.load_state_dict.assert_called_once_with(scheduler_state)
        self.assertTrue(any(update.kind == "info" for update in updates))


if __name__ == "__main__":
    unittest.main()
