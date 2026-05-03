import torch
from torch import nn
from transformers import Qwen3Model, Wav2Vec2Model


class LLMForcedAligner(nn.Module):
    # 320-sample stride / 16 000 Hz × 1 000
    FRAME_STRIDE_MS: float = 20.0

    def __init__(
        self,
        *,
        encoder_model_name: str = "facebook/mms-300m",
        llm_model_name: str = "Qwen/Qwen3-0.6B-Base",
        out_dim: int = 15_000,  # 300 s / 20 ms stride
        timestamp_token_id: int = 109,  # ±
    ) -> None:
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(encoder_model_name)
        self.llm = Qwen3Model.from_pretrained(llm_model_name)

        audio_dim = self.encoder.config.hidden_size
        llm_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(audio_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        self.head = nn.Linear(llm_dim, out_dim)

        self.timestamp_token_id = timestamp_token_id

    def forward(
        self,
        audio_input_values: torch.Tensor,  # (B, T_samples) raw waveform at 16 kHz
        text_input_ids: torch.Tensor,  # (B, S)
    ) -> torch.Tensor:
        # B         – batch size
        # T_samples – waveform length in samples (e.g. 4 800 000 for 300 s @ 16 kHz)
        # F         – encoder output frames (T_samples / 320, one frame per 20 ms)
        # d         – MMS encoder hidden size
        # (B, F, d)
        enc_outputs = self.encoder(audio_input_values)
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

        # (B, F + S, D)
        llm_outputs = self.llm(inputs_embeds=combined)
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
