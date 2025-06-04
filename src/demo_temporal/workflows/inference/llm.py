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


async def evaluate_model(inference_input: InferenceInput) -> dict:
    """
    Evaluate the model with the given parameters.
    Adapted from https://github.com/lantunes/CrystaLLM
    """
    torch.manual_seed(inference_input.seed)
    torch.cuda.manual_seed(inference_input.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in inference_input.device else "cpu"
    )  # for later use in torch.autocast
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[inference_input.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    checkpoint = torch.load(
        inference_input.model_path, map_location=inference_input.device
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
    model.to(inference_input.device)
    if inference_input.compile:
        model = torch.compile(model)

    # encode the beginning of the prompt
    prompt = inference_input.raw_input
    start_ids = encode(tokenizer.tokenize_cif(prompt))
    x = torch.tensor(start_ids, dtype=torch.long, device=inference_input.device)[
        None, ...
    ]

    # run generation
    generated = []
    with torch.no_grad():
        with ctx:
            for k in range(inference_input.num_samples):
                y = model.generate(
                    x,
                    inference_input.max_new_tokens,
                    temperature=inference_input.temperature,
                    top_k=inference_input.top_k,
                )
                generated.append(decode(y[0].tolist()))

    return {
        "prompt": prompt,
        "num_samples_generated": inference_input.num_samples,
        "samples": generated,
    }

