import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from .lightning import KaraokeAlignementsDataModule, LLMForcedAlignerLightning


class LLMForcedAlignerCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("model.encoder_checkpoint", "data.encoder_checkpoint")
        parser.link_arguments("model.llm_checkpoint", "data.llm_checkpoint")
        parser.link_arguments("model.max_duration", "data.max_duration")
        parser.link_arguments("model.timestamp_token_id", "data.timestamp_token_id")
        parser.link_arguments(
            "model.model",
            "data.frame_stride_ms",
            compute_fn=lambda model: model.frame_stride_ms,
            apply_on="instantiate",
        )
        parser.link_arguments(
            "model.model",
            "data.out_dim",
            compute_fn=lambda model: model.out_dim,
            apply_on="instantiate",
        )


def cli_main():
    torch.set_float32_matmul_precision("high")
    _ = LLMForcedAlignerCLI(LLMForcedAlignerLightning, KaraokeAlignementsDataModule)


if __name__ == "__main__":
    cli_main()
