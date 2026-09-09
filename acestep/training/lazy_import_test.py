"""Regression coverage for inference access to training configuration."""

import subprocess
import sys
import unittest
from pathlib import Path


class LazyTrainingImportTests(unittest.TestCase):
    """Keep configuration imports usable without loading training dependencies."""

    def test_adapter_config_does_not_load_training_stack(self) -> None:
        """Construct adapter configuration with heavyweight imports unavailable."""
        code = """
import importlib.abc
import sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'lightning', 'tensorboard', 'bitsandbytes'}:
            raise RuntimeError(f'Unexpected training dependency: {fullname}')
sys.meta_path.insert(0, BlockTraining())
from acestep.training.configs import LoKRConfig
from acestep.training import LoRAConfig
assert LoKRConfig(linear_dim=16).to_dict()['linear_dim'] == 16
assert LoRAConfig(r=4).to_dict()['r'] == 4
import acestep.training
assert not hasattr(acestep.training, 'unknown_training_api')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
