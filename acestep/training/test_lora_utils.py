"""Unit tests for acestep.training.lora_utils module."""

import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn

from acestep.training.lora_utils import (
    _unwrap_decoder,
    check_peft_available,
)


class TestUnwrapDecoder(unittest.TestCase):
    """Test cases for _unwrap_decoder function."""

    def test_returns_module_directly(self):
        """If module has no wrappers, return it unchanged."""
        mock_module = MagicMock(spec=nn.Module)
        result = _unwrap_decoder(mock_module)
        self.assertIs(result, mock_module)

    def test_unwraps_forward_module(self):
        """Unwrap _forward_module chain."""
        inner = MagicMock(spec=nn.Module)
        wrapper = MagicMock()
        wrapper._forward_module = inner

        result = _unwrap_decoder(wrapper)
        self.assertIs(result, inner)

    def test_unwraps_peft_base_model(self):
        """Unwrap PEFT base_model with .model attribute."""
        inner = MagicMock(spec=nn.Module)
        base = MagicMock()
        base.model = inner

        result = _unwrap_decoder(base)
        self.assertIs(result, inner)

    def test_unwraps_peft_base_model_no_inner_model(self):
        """Unwrap PEFT base_model without .model attribute."""
        base = MagicMock(spec=nn.Module)

        result = _unwrap_decoder(base)
        self.assertIs(result, base)

    def test_unwraps_nested_wrappers(self):
        """Handle multiple wrapper layers."""
        inner = MagicMock(spec=nn.Module)
        wrapper1 = MagicMock()
        wrapper1._forward_module = inner
        wrapper2 = MagicMock()
        wrapper2._forward_module = wrapper1

        result = _unwrap_decoder(wrapper2)
        self.assertIs(result, inner)

    def test_unwraps_complex_peft_chain(self):
        """Unwrap complex chain: wrapper -> _forward_module -> base_model -> .model."""
        inner = MagicMock(spec=nn.Module)
        base = MagicMock()
        base.model = inner
        mid = MagicMock()
        mid._forward_module = base
        wrapper = MagicMock()
        wrapper._forward_module = mid

        result = _unwrap_decoder(wrapper)
        self.assertIs(result, inner)


class TestCheckPeftAvailable(unittest.TestCase):
    """Test cases for check_peft_available function."""

    def test_returns_boolean(self):
        """check_peft_available should return a boolean."""
        result = check_peft_available()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
