import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from transformers import Mistral3ForConditionalGeneration


def fetch_transformer():
    """
    Fetches the transformer of the Flux.2 quantised to 4 bit
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return Flux2Transformer2DModel.from_pretrained(
            "diffusers/FLUX.2-dev-bnb-4bit",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=False,
        )
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}..."
                )
            else:
                raise

def fetch_text_encoder():
    """
    Fetches the text encoder of the Flux.2 quantised to 4 bit
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return Mistral3ForConditionalGeneration.from_pretrained(
            "diffusers/FLUX.2-dev-bnb-4bit",
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            device_map="cpu",
            load_in_4bit=True,
            local_files_only=False,
        )
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}..."
                )
            else:
                raise

def fetch_pipe(transformer, text_encoder):
    """
    Fetches the whole Flux 2 pipeline
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return Flux2Pipeline.from_pretrained(
            "diffusers/FLUX.2-dev-bnb-4bit",
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
            local_files_only=False,
        )
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}..."
                )
            else:
                raise


def get_flux2_pipeline():
    """
    Fetches the 4bit Flux.2 pipeline from the HuggingFace model hub.
    """
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True,
    }

    transformer = fetch_transformer()
    text_encoder = fetch_text_encoder()
    pipe = fetch_pipe(transformer, text_encoder)

    return pipe, transformer, text_encoder


if __name__ == "__main__":
    get_flux2_pipeline()
