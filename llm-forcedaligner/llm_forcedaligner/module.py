from typing import TypedDict, Unpack, cast

import torch
from torch import nn
from transformers import Qwen3Model, Wav2Vec2Model


class ModelInput(TypedDict):
    audio_input_values: torch.Tensor
    audio_attention_mask: torch.Tensor
    text_input_ids: torch.Tensor
    text_attention_mask: torch.Tensor


class LLMForcedAligner(nn.Module):
    def __init__(
        self,
        *,
        encoder_checkpoint: str,
        llm_checkpoint: str,
        max_duration: int,
        timestamp_token_id: int,
        attn_implementation: str,
    ) -> None:
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(
            encoder_checkpoint,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        self.llm = Qwen3Model.from_pretrained(
            llm_checkpoint,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )

        self.frame_stride_ms = (
            1000 / self.encoder._get_feat_extract_output_lengths(16000).item()  # pyright: ignore[reportAttributeAccessIssue]
        )

        self.encoder_dim = cast(int, self.encoder.config.output_hidden_size)
        self.llm_dim = self.llm.config.hidden_size
        self.out_dim = int(max_duration * 1000 / self.frame_stride_ms)

        self.projector = nn.Linear(self.encoder_dim, self.llm_dim, dtype=torch.bfloat16)
        self.head = nn.Linear(self.llm_dim, self.out_dim, dtype=torch.bfloat16)

        self.timestamp_token_id = timestamp_token_id

    def forward(self, /, **kwargs: Unpack[ModelInput]) -> torch.Tensor:
        audio_input_values = kwargs["audio_input_values"]
        audio_attention_mask = kwargs["audio_attention_mask"]
        text_input_ids = kwargs["text_input_ids"]
        text_attention_mask = kwargs["text_attention_mask"]

        enc_outputs = self.encoder(
            input_values=audio_input_values,
            attention_mask=audio_attention_mask,
        )
        enc_hidden = enc_outputs.last_hidden_state
        audio_embeds = self.projector(enc_hidden)

        text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        audio_input_lengths = audio_attention_mask.sum(-1)
        audio_output_lengths = self.encoder._get_feat_extract_output_lengths(
            audio_input_lengths  # pyright: ignore[reportArgumentType]
        )

        audio_max_frames = enc_hidden.size(1)
        audio_output_mask = (
            torch.arange(audio_max_frames, device=enc_hidden.device)
            < audio_output_lengths[:, None]  # pyright: ignore[reportIndexIssue]
        ).long()
        audio_positions = (audio_output_mask.cumsum(dim=1) - 1).clamp(min=0)

        text_positions = (
            audio_output_lengths[:, None] + text_attention_mask.cumsum(dim=1) - 1  # pyright: ignore[reportIndexIssue]
        ).clamp(min=0)

        combined = torch.cat([audio_embeds, text_embeds], dim=1)
        llm_attention_mask = torch.cat([audio_output_mask, text_attention_mask], dim=1)
        position_ids = torch.cat([audio_positions, text_positions], dim=1)

        llm_outputs = self.llm(
            inputs_embeds=combined,
            attention_mask=llm_attention_mask,
            position_ids=position_ids,
        )
        llm_hidden = llm_outputs.last_hidden_state

        text_hidden = llm_hidden[:, audio_embeds.size(1) :, :]
        slot_hidden = text_hidden[text_input_ids == self.timestamp_token_id]

        logits = self.head(slot_hidden)

        return logits
