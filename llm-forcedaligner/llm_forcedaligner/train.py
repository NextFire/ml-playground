import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from .lightning import KaraokeAlignementsDataModule, LLMForcedAlignerLightning


class LLMForcedAlignerCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("model.encoder_model_name", "data.encoder_model_name")
        parser.link_arguments("model.llm_model_name", "data.llm_model_name")
        parser.link_arguments("model.out_dim", "data.out_dim")
        parser.link_arguments("model.timestamp_token_id", "data.timestamp_token_id")


def cli_main():
    torch.set_float32_matmul_precision("high")
    _ = LLMForcedAlignerCLI(
        LLMForcedAlignerLightning,
        KaraokeAlignementsDataModule,
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    cli_main()
