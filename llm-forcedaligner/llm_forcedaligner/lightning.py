from functools import partial
from typing import Any, TypedDict

import lightning as L
import numpy as np
import torch
from datasets import Audio, Dataset, load_dataset
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoTokenizer

from .module import LLMForcedAligner


class ModelInput(TypedDict):
    audio_input_features: torch.Tensor  # (B, mel_T, mel_F)
    audio_attention_mask: torch.Tensor  # (B, mel_T)
    text_input_ids: torch.Tensor  # (B, S)
    text_attention_mask: torch.Tensor  # (B, S)
    labels: torch.Tensor  # (total_slots_in_batch,)


class LLMForcedAlignerLightning(L.LightningModule):
    def __init__(
        self,
        *,
        encoder_checkpoint: str = "google/gemma-4-E2B",
        llm_checkpoint: str = "Qwen/Qwen3-0.6B-Base",
        max_duration: int = 300,
        timestamp_token_id: int = 37021,  # ±
        attn_implementation: str = "sdpa",
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = LLMForcedAligner(
            encoder_checkpoint=encoder_checkpoint,
            llm_checkpoint=llm_checkpoint,
            max_duration=max_duration,
            timestamp_token_id=timestamp_token_id,
            attn_implementation=attn_implementation,
        )
        # self.model.encoder.gradient_checkpointing_enable()
        self.model.llm.gradient_checkpointing_enable()
        self.train()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,  # pyright: ignore[reportAttributeAccessIssue]
            weight_decay=self.hparams.weight_decay,  # pyright: ignore[reportAttributeAccessIssue]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / self.hparams.warmup_steps),  # pyright: ignore[reportAttributeAccessIssue]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, **kwargs) -> torch.Tensor:
        return self.model(**kwargs)

    def training_step(self, batch: ModelInput, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: ModelInput, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch: ModelInput, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="test")

    def _shared_step(
        self,
        batch: ModelInput,
        batch_idx: int,
        *,
        stage: str,
    ) -> torch.Tensor:
        # logits: (total_slots_in_batch, out_dim)
        logits = self(
            audio_input_features=batch["audio_input_features"],
            audio_attention_mask=batch["audio_attention_mask"],
            text_input_ids=batch["text_input_ids"],
            text_attention_mask=batch["text_attention_mask"],
        )
        # labels: (total_slots_in_batch,)
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        aas_frames = (preds.float() - labels.float()).abs().mean()
        aas_ms = aas_frames * self.model.frame_stride_ms
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_aas_frames", aas_frames, prog_bar=True)
        self.log(f"{stage}_aas_ms", aas_ms, prog_bar=True)
        return loss


class DatasetMora(TypedDict):
    value: str
    start: int
    end: int


class DatasetRow(TypedDict):
    audio: Audio
    morae: list[DatasetMora]


class KaraokeAlignementsDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        dataset_path: str,
        dataset_split: str,
        encoder_checkpoint: str,
        llm_checkpoint: str,
        max_duration: int,
        timestamp_token_id: int,
        frame_stride_ms: int,
        out_dim: int,
        load_from_cache_file: bool = True,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.encoder_checkpoint = encoder_checkpoint
        self.llm_checkpoint = llm_checkpoint
        self.max_duration = max_duration
        self.timestamp_token_id = timestamp_token_id
        self.frame_stride_ms = frame_stride_ms
        self.out_dim = out_dim
        self.load_from_cache_file = load_from_cache_file
        self.batch_size = batch_size
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            encoder_checkpoint
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_checkpoint)
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if all((self.train_dataset, self.val_dataset, self.test_dataset)):
            return
        dataset = load_dataset(self.dataset_path, split=self.dataset_split)
        dataset = dataset.map(
            lambda morae: {
                "morae": sorted(
                    [mora for mora in morae if mora["value"].strip()],
                    key=lambda mora: mora["start"],
                )
            },
            input_columns=["morae"],
            writer_batch_size=500,
            new_fingerprint="llmfa_sort_morae",
            load_from_cache_file=self.load_from_cache_file,
        )
        dataset = dataset.filter(
            lambda morae: (
                len(morae) > 0
                and all(
                    morae[i]["end"] <= morae[i + 1]["start"]
                    for i in range(len(morae) - 1)
                )
            ),
            input_columns=["morae"],
            new_fingerprint="llmfa_no_overlap",
            load_from_cache_file=self.load_from_cache_file,
        )
        dataset = dataset.filter(
            lambda audio: audio.metadata.duration_seconds <= self.max_duration,
            input_columns=["audio"],
            new_fingerprint=f"llmfa_max_duration_{self.max_duration}",
            load_from_cache_file=self.load_from_cache_file,
        )
        split = dataset.train_test_split(test_size=0.15)
        self.train_dataset = split["train"]
        split = split["test"].train_test_split(test_size=0.5)
        self.val_dataset = split["train"]
        self.test_dataset = split["test"]

    def train_dataloader(self) -> Any:
        assert self.train_dataset
        return DataLoader(
            self.train_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=partial(self._collate_batch, dynamic_slot_prob=0.5),
            shuffle=True,
        )

    def val_dataloader(self) -> Any:
        assert self.val_dataset
        return DataLoader(
            self.val_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=partial(self._collate_batch, dynamic_slot_prob=0.0),
        )

    def test_dataloader(self) -> Any:
        assert self.test_dataset
        return DataLoader(
            self.test_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=partial(self._collate_batch, dynamic_slot_prob=0.0),
        )

    def _collate_batch(
        self,
        batch: list[DatasetRow],
        *,
        dynamic_slot_prob: float = 0.0,
    ) -> ModelInput:
        audio_arrays: list[np.ndarray] = []
        sampling_rates: list[int] = []
        all_token_ids: list[list[int]] = []
        all_labels: list[torch.Tensor] = []
        for example in batch:
            audio = example["audio"]
            audio_arrays.append(audio["array"])  # pyright: ignore[reportIndexIssue]
            sampling_rates.append(audio["sampling_rate"])  # pyright: ignore[reportIndexIssue]
            token_ids, label_ids = self._build_text_and_labels(
                example["morae"],
                dynamic_slot_prob=dynamic_slot_prob,
            )
            all_token_ids.append(token_ids)
            all_labels.append(torch.tensor(label_ids))
        fe_out = self.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rates,
            truncation=False,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        # (B, S) — tokenizer.pad() handles padding_value and returns attention_mask
        text_padded = self.tokenizer.pad(
            [{"input_ids": token_ids} for token_ids in all_token_ids],
            return_tensors="pt",
        )
        # Labels have variable slot count per sample; concatenate to align with
        # the flattened slot_hidden produced by the model forward.
        # (total_slots_in_batch,)
        labels = torch.cat(all_labels)
        return {
            "audio_input_features": fe_out["input_features"],
            "audio_attention_mask": fe_out["input_features_mask"],
            "text_input_ids": text_padded["input_ids"],
            "text_attention_mask": text_padded["attention_mask"],
            "labels": labels,
        }

    def _build_text_and_labels(
        self,
        morae: list[DatasetMora],
        *,
        dynamic_slot_prob: float,
    ) -> tuple[list[int], list[int]]:
        apply_dynamic = np.random.random() < dynamic_slot_prob

        token_ids: list[int] = []
        label_ids: list[int] = []

        for mora in morae:
            mora_ids = self.tokenizer.encode(mora["value"], add_special_tokens=False)
            token_ids.extend(mora_ids)
            # Dynamic slot insertion: always insert when dynamic is off; otherwise 50% chance
            if (not apply_dynamic) or (np.random.random() < 0.5):
                # Discretise
                start_idx = mora["start"] // self.frame_stride_ms
                end_idx = min(mora["end"] // self.frame_stride_ms, self.out_dim - 1)
                # Start slot
                token_ids.append(self.timestamp_token_id)
                label_ids.append(start_idx)
                # End slot
                token_ids.append(self.timestamp_token_id)
                label_ids.append(end_idx)

        if apply_dynamic and len(label_ids) == 0:
            return self._build_text_and_labels(morae, dynamic_slot_prob=0.0)

        return token_ids, label_ids
