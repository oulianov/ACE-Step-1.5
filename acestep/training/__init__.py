"""Training APIs loaded on demand so inference can import adapter configuration cheaply."""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "DatasetBuilder": "acestep.training.dataset_builder",
    "AudioSample": "acestep.training.dataset_builder",
    "LoRAConfig": "acestep.training.configs",
    "LoKRConfig": "acestep.training.configs",
    "TrainingConfig": "acestep.training.configs",
    "inject_lora_into_dit": "acestep.training.lora_injection",
    "freeze_non_lora_parameters": "acestep.training.lora_injection",
    "save_lora_weights": "acestep.training.lora_checkpoint",
    "load_lora_weights": "acestep.training.lora_checkpoint",
    "save_training_checkpoint": "acestep.training.lora_checkpoint",
    "load_training_checkpoint": "acestep.training.lora_checkpoint",
    "merge_lora_weights": "acestep.training.lora_utils",
    "check_peft_available": "acestep.training.lora_utils",
    "inject_lokr_into_dit": "acestep.training.lokr_utils",
    "save_lokr_weights": "acestep.training.lokr_utils",
    "load_lokr_weights": "acestep.training.lokr_utils",
    "check_lycoris_available": "acestep.training.lokr_utils",
    "PreprocessedTensorDataset": "acestep.training.data_module",
    "PreprocessedDataModule": "acestep.training.data_module",
    "collate_preprocessed_batch": "acestep.training.data_module",
    "AceStepTrainingDataset": "acestep.training.data_module",
    "AceStepDataModule": "acestep.training.data_module",
    "collate_training_batch": "acestep.training.data_module",
    "load_dataset_from_json": "acestep.training.data_module",
    "LoRATrainer": "acestep.training.trainer",
    "LoKRTrainer": "acestep.training.trainer",
    "PreprocessedLoRAModule": "acestep.training.trainer",
    "PreprocessedLoKRModule": "acestep.training.trainer",
    "LIGHTNING_AVAILABLE": "acestep.training.trainer",
}


def __getattr__(name: str) -> Any:
    """Resolve public training APIs only when requested; reject unknown attributes."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def check_lightning_available() -> bool:
    """Check Lightning availability when a caller requests training support."""
    return import_module("acestep.training.trainer").LIGHTNING_AVAILABLE


__all__ = [
    "DatasetBuilder",
    "AudioSample",
    "LoRAConfig",
    "LoKRConfig",
    "TrainingConfig",
    "inject_lora_into_dit",
    "freeze_non_lora_parameters",
    "save_lora_weights",
    "load_lora_weights",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "merge_lora_weights",
    "check_peft_available",
    "inject_lokr_into_dit",
    "save_lokr_weights",
    "load_lokr_weights",
    "check_lycoris_available",
    "PreprocessedTensorDataset",
    "PreprocessedDataModule",
    "collate_preprocessed_batch",
    "AceStepTrainingDataset",
    "AceStepDataModule",
    "collate_training_batch",
    "load_dataset_from_json",
    "LoRATrainer",
    "LoKRTrainer",
    "PreprocessedLoRAModule",
    "PreprocessedLoKRModule",
    "check_lightning_available",
    "LIGHTNING_AVAILABLE",
]
