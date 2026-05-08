from typing import cast

import torch
from torch import nn
from transformers import Gemma4AudioConfig, Gemma4AudioModel, Qwen3Model
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioSubSampleConvProjection,
    Gemma4AudioSubSampleConvProjectionLayer,
)
from transformers.monkey_patching import register_patch_mapping


class CustomGemma4AudioSubSampleConvProjection(Gemma4AudioSubSampleConvProjection):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__(config)
        self.layer0 = CustomGemma4AudioSubSampleConvProjectionLayer(
            in_channels=1,
            out_channels=config.subsampling_conv_channels[0],
            norm_eps=config.rms_norm_eps,
            temporal_stride=1,
            frequential_stride=2,
        )
        self.layer1 = CustomGemma4AudioSubSampleConvProjectionLayer(
            in_channels=config.subsampling_conv_channels[0],
            out_channels=config.subsampling_conv_channels[1],
            norm_eps=config.rms_norm_eps,
            temporal_stride=2,
            frequential_stride=2,
        )


class CustomGemma4AudioSubSampleConvProjectionLayer(
    Gemma4AudioSubSampleConvProjectionLayer
):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_eps,
        temporal_stride: int = 2,
        frequential_stride: int = 2,
    ):
        super().__init__(in_channels, out_channels, norm_eps)
        self.temporal_stride = temporal_stride
        self.frequential_stride = frequential_stride
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(temporal_stride, frequential_stride),
            padding=1,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is not None:
            mask = mask.to(device=hidden_states.device)
            hidden_states = hidden_states * mask[:, None, :, None]

        hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
        hidden_states = self.act(
            self.norm(hidden_states.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        if mask is not None and self.temporal_stride > 1:
            mask = mask[:, :: self.temporal_stride]

        return hidden_states, mask


class LLMForcedAligner(nn.Module):
    def __init__(
        self,
        *,
        encoder_checkpoint: str,
        encoder_num_hidden_layers: int,
        llm_checkpoint: str,
        max_duration: int,
        timestamp_token_id: int,
        attn_implementation: str,
    ) -> None:
        super().__init__()

        register_patch_mapping(
            {
                Gemma4AudioSubSampleConvProjection.__name__: CustomGemma4AudioSubSampleConvProjection,
                Gemma4AudioSubSampleConvProjectionLayer.__name__: CustomGemma4AudioSubSampleConvProjectionLayer,
            },
            overwrite=True,
        )
        self.encoder = Gemma4AudioModel.from_pretrained(
            encoder_checkpoint,
            config=Gemma4AudioConfig(num_hidden_layers=encoder_num_hidden_layers),
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

        proj = cast(
            CustomGemma4AudioSubSampleConvProjection,
            self.encoder.subsample_conv_projection,
        )
        self.temporal_stride = proj.layer0.temporal_stride * proj.layer1.temporal_stride
        self.frame_stride_ms = 10 * self.temporal_stride

        self.encoder_dim = self.encoder.config.output_proj_dims
        self.llm_dim = self.llm.config.hidden_size
        self.out_dim = max_duration * 1000 // self.frame_stride_ms

        self.projector = nn.Linear(self.encoder_dim, self.llm_dim)
        self.head = nn.Linear(self.llm_dim, self.out_dim)

        self.timestamp_token_id = timestamp_token_id

    def forward(
        self,
        *,
        audio_input_features: torch.Tensor,  # (B, mel_T, mel_F) spectrogram features
        audio_attention_mask: torch.Tensor,  # (B, mel_T) from feature extractor
        text_input_ids: torch.Tensor,  # (B, S)
        text_attention_mask: torch.Tensor,  # (B, S) from tokenizer.pad()
    ) -> torch.Tensor:
        # B – batch size
        # F – encoder output frames (= mel_T because temporal stride is 1)
        # d – encoder hidden size
        # (B, F, d)
        enc_outputs = self.encoder(
            input_features=audio_input_features,
            attention_mask=audio_attention_mask,
        )
        enc_hidden = enc_outputs.last_hidden_state

        # D – LLM hidden size
        # (B, F, D)
        audio_embeds = self.projector(enc_hidden)

        # S – text sequence length (mora tokens + ± slot tokens)
        # (B, S, D)
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        # Prepend audio embeddings to text embeddings as the LLM context.
        # (B, F + S, D)
        combined = torch.cat([audio_embeds, text_embeds], dim=1)

        # Build attention mask for the LLM:
        # - audio positions: downsample audio_attention_mask (B, mel_T) to match
        #   encoder output length F = mel_T / audio_temporal_stride.
        # - text positions: use attention_mask produced by tokenizer.pad().
        # (B, F+S)
        F = audio_embeds.size(1)
        audio_mask_for_llm = audio_attention_mask[:, :: self.temporal_stride][:, :F]
        llm_attention_mask = torch.cat(
            [audio_mask_for_llm, text_attention_mask],
            dim=1,
        )

        # (B, F + S, D)
        llm_outputs = self.llm(
            inputs_embeds=combined,
            attention_mask=llm_attention_mask,
        )
        llm_hidden = llm_outputs.last_hidden_state

        # Slice out only the text positions from the joint sequence.
        # (B, S, D)
        text_hidden = llm_hidden[:, audio_embeds.size(1) :, :]

        # Keep only positions where the slot token (±) appears.
        # (total_slots_in_batch, D)
        slot_hidden = text_hidden[text_input_ids == self.timestamp_token_id]

        # (total_slots_in_batch, out_dim)  — distribution over timestamp classes
        logits = self.head(slot_hidden)

        return logits
