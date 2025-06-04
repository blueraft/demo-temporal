from contextlib import nullcontext
import os

import asyncio
import aiohttp
from crystallm import (
    CIFTokenizer,
    GPT,
    GPTConfig,
)
import torch

from demo_temporal.workflows.shared import InferenceInput

BLOCK_SIZE = 1024


async def download_model(model_path: str, model_url: str | None = None) -> dict:
    """
    Checks if the model file exists locally, and if not, downloads it from the
    provided URL.
    """
    # Check if file exists asynchronously
    exists = await asyncio.to_thread(os.path.exists, model_path)
    if not exists and not model_url:
        raise FileNotFoundError(
            f'Model file "{model_path}" does not exist and `model_url` is not provided.'
        )
    elif exists and model_url:
        return {
            "model_path": model_path,
            "model_url": model_url,
        }
    elif exists:
        return {"model_path": model_path}

    # Download the model from the URL asynchronously
    async with aiohttp.ClientSession() as session:
        async with session.get(model_url) as response:
            if response.status != 200:
                raise ValueError(f'Failed to download model from "{model_url}".')
            # Download in chunks
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            loop = asyncio.get_running_loop()
            with open(model_path, "wb") as f:
                async for chunk in response.content.iter_chunked(BLOCK_SIZE):
                    await loop.run_in_executor(None, f.write, chunk)

    return {"model_path": model_path, "model_url": model_url}


async def evaluate_model(inference_state: InferenceInput) -> dict:
    """
    Evaluate the model with the given parameters.
    Adapted from https://github.com/lantunes/CrystaLLM
    """
    torch.manual_seed(inference_state.seed)
    torch.cuda.manual_seed(inference_state.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in inference_state.device else "cpu"
    )  # for later use in torch.autocast
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[inference_state.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    checkpoint = torch.load(
        inference_state.model_path, map_location=inference_state.device
    )
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(inference_state.device)
    if inference_state.compile:
        model = torch.compile(model)

    # encode the beginning of the prompt
    prompt = inference_state.raw_input
    start_ids = encode(tokenizer.tokenize_cif(prompt))
    x = torch.tensor(start_ids, dtype=torch.long, device=inference_state.device)[
        None, ...
    ]

    # run generation
    generated = []
    with torch.no_grad():
        with ctx:
            for k in range(inference_state.num_samples):
                y = model.generate(
                    x,
                    inference_state.max_new_tokens,
                    temperature=inference_state.temperature,
                    top_k=inference_state.top_k,
                )
                generated.append(decode(y[0].tolist()))

    inference_state.generated_samples = generated

    return inference_state


async def write_cif_files(
    inference_input: InferenceInput, generated: list[str]
) -> None:
    """
    Write the generated CIFs to the specified target (console or file).
    """
    if inference_input.generate_cif:
        for k, sample in enumerate(generated):
            fname = f"sample_{k + 1}.cif"
            with open(fname, "wt", encoding="utf-8") as f:
                f.write(sample)
