"""
PyTorch Lightning DataModule for LoRA Training

Handles data loading and preprocessing for training ACE-Step LoRA adapters.
Supports both raw audio loading and preprocessed tensor loading.
"""

import os
import json
import random
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger

from acestep.training.path_safety import safe_path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

try:
    from lightning.pytorch import LightningDataModule
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    logger.warning("Lightning not installed. Training module will not be available.")
    # Create a dummy class for type hints
    class LightningDataModule:
        pass


# ============================================================================
# Preprocessed Tensor Dataset (Recommended for Training)
# ============================================================================

from acestep.training.preprocessed_sampler import BucketedBatchSampler
from acestep.training.preprocessed_dataset import PreprocessedTensorDataset
from acestep.training.preprocessed_collate import collate_preprocessed_batch


class PreprocessedDataModule(LightningDataModule if LIGHTNING_AVAILABLE else object):
    """DataModule for preprocessed tensor files.

    Loads precomputed tensors directly, avoiding VAE/text encoding at train time.
    """

    def __init__(
        self,
        tensor_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory_device: str = "",
        val_split: float = 0.0,
        length_bucket: bool = False,
        cache_policy: str = "none",
        cache_max_items: int = 0,
    ):
        """Initialize the preprocessed data module.

        Args:
            tensor_dir: Directory containing preprocessed ``.pt`` files.
            batch_size: Training batch size.
            num_workers: Number of DataLoader worker processes.
            pin_memory: Pin host memory for faster GPU transfer.
            prefetch_factor: Number of prefetched batches per worker.
            persistent_workers: Keep worker processes alive between epochs.
            pin_memory_device: Device string used by pinned memory allocator.
            val_split: Fraction of data reserved for validation.
            length_bucket: Whether to bucket training samples by latent length.
            cache_policy: Dataset cache mode ("none" or "ram_lru").
            cache_max_items: Maximum cached entries when RAM LRU is enabled.
        """
        if LIGHTNING_AVAILABLE:
            super().__init__()

        self.tensor_dir = tensor_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device
        self.val_split = val_split
        self.length_bucket = length_bucket
        self.cache_policy = cache_policy
        self.cache_max_items = cache_max_items

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup training/validation datasets."""
        if stage == "fit" or stage is None:
            full_dataset = PreprocessedTensorDataset(
                self.tensor_dir,
                cache_policy=self.cache_policy,
                cache_max_items=self.cache_max_items,
            )
            if self.val_split > 0 and len(full_dataset) > 1:
                n_val = max(1, int(len(full_dataset) * self.val_split))
                n_train = len(full_dataset) - n_val
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_dataset, [n_train, n_val]
                )
            else:
                self.train_dataset = full_dataset
                self.val_dataset = None

    def _resolve_train_latent_lengths(self) -> Optional[List[int]]:
        """Resolve latent lengths for bucketed sampling, including Subset splits."""
        if not self.length_bucket or self.train_dataset is None:
            return None

        ds = self.train_dataset
        if isinstance(ds, torch.utils.data.Subset):
            base = ds.dataset
            indices = list(ds.indices)
            base_lengths = getattr(base, "latent_lengths", None)
            if base_lengths is None:
                return None
            return [base_lengths[i] for i in indices]

        base_lengths = getattr(ds, "latent_lengths", None)
        if base_lengths is None:
            return None
        return list(base_lengths)

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        prefetch_factor = None if self.num_workers == 0 else self.prefetch_factor
        persistent_workers = False if self.num_workers == 0 else self.persistent_workers
        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.length_bucket,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_preprocessed_batch,
            drop_last=False,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        if self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device

        latent_lengths = self._resolve_train_latent_lengths()
        if latent_lengths is not None:
            kwargs.pop("batch_size", None)
            kwargs.pop("shuffle", None)
            kwargs["batch_sampler"] = BucketedBatchSampler(
                lengths=latent_lengths,
                batch_size=self.batch_size,
                shuffle=True,
            )
        elif self.length_bucket:
            kwargs["shuffle"] = True

        return DataLoader(**kwargs)

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None
        prefetch_factor = None if self.num_workers == 0 else self.prefetch_factor
        persistent_workers = False if self.num_workers == 0 else self.persistent_workers
        kwargs = dict(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_preprocessed_batch,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        if self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return DataLoader(**kwargs)


# ============================================================================
# Raw Audio Dataset (Legacy - for backward compatibility)
# ============================================================================

class AceStepTrainingDataset(Dataset):
    """Dataset for ACE-Step LoRA training from raw audio.
    
    DEPRECATED: Use PreprocessedTensorDataset instead for better performance.
    
    Audio Format Requirements (handled automatically):
    - Sample rate: 48kHz (resampled if different)
    - Channels: Stereo (2 channels, mono is duplicated)
    - Max duration: 240 seconds (4 minutes)
    - Min duration: 5 seconds (padded if shorter)
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dit_handler,
        max_duration: float = 240.0,
        target_sample_rate: int = 48000,
    ):
        """Initialize the dataset."""
        self.samples = samples
        self.dit_handler = dit_handler
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        
        self.valid_samples = self._validate_samples()
        logger.info(f"Dataset initialized with {len(self.valid_samples)} valid samples")
    
    def _validate_samples(self) -> List[Dict[str, Any]]:
        """Validate and filter samples, resolving audio paths to safe paths."""
        valid = []
        for i, sample in enumerate(self.samples):
            audio_path = sample.get("audio_path", "")
            if not audio_path:
                logger.warning(f"Sample {i}: Missing audio_path")
                continue

            try:
                validated = safe_path(audio_path)
            except ValueError:
                logger.warning(f"Sample {i}: Rejected unsafe path: {audio_path}")
                continue

            if not os.path.isfile(validated):
                logger.warning(f"Sample {i}: Audio file not found: {audio_path}")
                continue
            
            if not sample.get("caption"):
                logger.warning(f"Sample {i}: Missing caption")
                continue
            
            # Store validated path so downstream code never uses raw user input
            sample = {**sample, "audio_path": validated}
            valid.append(sample)
        
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        sample = self.valid_samples[idx]
        
        audio_path = sample["audio_path"]
        audio, sr = torchaudio.load(audio_path)
        
        # Resample to 48kHz
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        
        # Convert to stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        
        # Truncate/pad
        max_samples = int(self.max_duration * self.target_sample_rate)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        
        min_samples = int(5.0 * self.target_sample_rate)
        if audio.shape[1] < min_samples:
            padding = min_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        return {
            "audio": audio,
            "caption": sample.get("caption", ""),
            "lyrics": sample.get("lyrics", "[Instrumental]"),
            "metadata": {
                "caption": sample.get("caption", ""),
                "lyrics": sample.get("lyrics", "[Instrumental]"),
                "bpm": sample.get("bpm"),
                "keyscale": sample.get("keyscale", ""),
                "timesignature": sample.get("timesignature", ""),
                "duration": sample.get("duration", audio.shape[1] / self.target_sample_rate),
                "language": sample.get("language", "unknown"),
                "is_instrumental": sample.get("is_instrumental", True),
            },
            "audio_path": audio_path,
        }


def collate_training_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for raw audio batches (legacy)."""
    max_len = max(sample["audio"].shape[1] for sample in batch)
    
    padded_audio = []
    attention_masks = []
    
    for sample in batch:
        audio = sample["audio"]
        audio_len = audio.shape[1]
        
        if audio_len < max_len:
            padding = max_len - audio_len
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        padded_audio.append(audio)
        
        mask = torch.ones(max_len)
        if audio_len < max_len:
            mask[audio_len:] = 0
        attention_masks.append(mask)
    
    return {
        "audio": torch.stack(padded_audio),
        "attention_mask": torch.stack(attention_masks),
        "captions": [s["caption"] for s in batch],
        "lyrics": [s["lyrics"] for s in batch],
        "metadata": [s["metadata"] for s in batch],
        "audio_paths": [s["audio_path"] for s in batch],
    }


class AceStepDataModule(LightningDataModule if LIGHTNING_AVAILABLE else object):
    """DataModule for raw audio loading (legacy).
    
    DEPRECATED: Use PreprocessedDataModule for better training performance.
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dit_handler,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_duration: float = 240.0,
        val_split: float = 0.0,
        length_bucket: bool = False,
        cache_policy: str = "none",
        cache_max_items: int = 0,
    ):
        """Initialize legacy raw-audio datamodule.

        Args:
            samples: Raw training sample metadata entries.
            dit_handler: Model handler used by legacy training flows.
            batch_size: Number of samples per batch.
            num_workers: Number of dataloader workers.
            pin_memory: Whether to enable pinned memory in dataloaders.
            max_duration: Max audio duration (seconds) for clipping.
            val_split: Validation split fraction.
            length_bucket: Accepted for compatibility; unused for raw audio mode.
            cache_policy: Accepted for compatibility; unused for raw audio mode.
            cache_max_items: Accepted for compatibility; unused for raw audio mode.
        """
        if LIGHTNING_AVAILABLE:
            super().__init__()

        self.samples = samples
        self.dit_handler = dit_handler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_duration = max_duration
        self.val_split = val_split
        self.length_bucket = length_bucket
        self.cache_policy = cache_policy
        self.cache_max_items = cache_max_items
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            if self.val_split > 0 and len(self.samples) > 1:
                n_val = max(1, int(len(self.samples) * self.val_split))
                
                indices = list(range(len(self.samples)))
                random.shuffle(indices)
                
                val_indices = indices[:n_val]
                train_indices = indices[n_val:]
                
                train_samples = [self.samples[i] for i in train_indices]
                val_samples = [self.samples[i] for i in val_indices]
                
                self.train_dataset = AceStepTrainingDataset(
                    train_samples, self.dit_handler, self.max_duration
                )
                self.val_dataset = AceStepTrainingDataset(
                    val_samples, self.dit_handler, self.max_duration
                )
            else:
                self.train_dataset = AceStepTrainingDataset(
                    self.samples, self.dit_handler, self.max_duration
                )
                self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_training_batch,
            drop_last=True,
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_training_batch,
        )


def load_dataset_from_json(json_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load a dataset from JSON file.

    Args:
        json_path: Path to the JSON dataset file.

    Returns:
        Tuple of (samples list, metadata dict).

    Raises:
        ValueError: If json_path does not point to an existing file or escapes safe root.
    """
    validated = safe_path(json_path)
    if not os.path.isfile(validated):
        raise ValueError(f"Dataset JSON file not found: {json_path}")

    with open(validated, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    samples = data.get("samples", [])
    
    return samples, metadata
