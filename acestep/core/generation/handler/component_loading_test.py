"""Exercise component loading against real, small serialized model checkpoints."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from diffusers import AutoencoderOobleck
from transformers import AutoModel, Qwen3Config, Qwen3Model

spec = importlib.util.spec_from_file_location(
    "component_loader", Path(__file__).with_name("init_service_loader_components.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComponentHost(module.InitServiceLoaderComponentsMixin):
    """Use CPU execution to verify checkpoint values and offload dtype choices."""

    dtype = torch.bfloat16
    offload_to_cpu = False

    def _get_vae_dtype(self, device: str) -> torch.dtype:
        """Return the production CPU-safe VAE dtype."""
        return torch.float32


class ComponentLoadingTests(unittest.TestCase):
    """Compare newly loaded components with the previous load-then-cast behavior."""

    def test_text_encoder_checkpoint_preserves_outputs_and_offload_dtype(self) -> None:
        """Load real Qwen weights in BF16 for inference or FP32 for CPU offload."""
        torch.manual_seed(51)
        model = Qwen3Model(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
            )
        ).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Qwen3-Embedding-0.6B"
            model.save_pretrained(path)
            for offload in (False, True):
                with self.subTest(offload=offload):
                    host = ComponentHost()
                    host.offload_to_cpu = offload
                    expected_dtype = torch.float32 if offload else host.dtype
                    reference = AutoModel.from_pretrained(path).to(expected_dtype).eval()
                    with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()):
                        host._load_text_encoder_and_tokenizer(
                            checkpoint_dir=directory, device="cpu"
                        )
                    self.assertEqual(next(host.text_encoder.parameters()).dtype, expected_dtype)
                    self.assertFalse(host.text_encoder.training)
                    with torch.no_grad():
                        tokens = torch.tensor([[1, 4, 9]])
                        actual = host.text_encoder(tokens).last_hidden_state
                        expected = reference(tokens).last_hidden_state
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_vae_checkpoint_preserves_decoded_audio(self) -> None:
        """The dtype-aware VAE loader preserves waveform output on the CPU path."""
        torch.manual_seed(52)
        model = AutoencoderOobleck(
            encoder_hidden_size=4,
            downsampling_ratios=[2, 2],
            channel_multiples=[1, 2],
            decoder_channels=4,
            decoder_input_channels=2,
            audio_channels=2,
        ).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vae"
            model.save_pretrained(path)
            reference = AutoencoderOobleck.from_pretrained(path).to(torch.float32).eval()
            host = ComponentHost()
            host._load_vae_model(checkpoint_dir=directory, device="cpu", compile_model=False)
            with torch.no_grad():
                latents = torch.randn(1, 2, 32)
                actual = host.vae.decode(latents).sample
                expected = reference.decode(latents).sample
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(host.vae.training)


if __name__ == "__main__":
    unittest.main()
