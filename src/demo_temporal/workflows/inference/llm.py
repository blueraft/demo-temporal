from contextlib import nullcontext
import os
import shutil
import tarfile
import tempfile

import asyncio
import aiohttp
from crystallm import (
    CIFTokenizer,
    GPT,
    GPTConfig,
)
import torch

from demo_temporal.workflows.shared import InferenceModelInput, InferenceResultsInput

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

    # Download the model from the URL and copy the model file to the model_path
    with tempfile.TemporaryDirectory() as tmpdir:
        async with aiohttp.ClientSession() as session:
            async with session.get(model_url) as response:
                if response.status != 200:
                    raise ValueError(f'Failed to download model from "{model_url}".')
                # Download in chunks
                tmp_zipfile = os.path.join(tmpdir, model_url.split("/")[-1])
                loop = asyncio.get_running_loop()
                with open(tmp_zipfile, "wb") as f:
                    async for chunk in response.content.iter_chunked(BLOCK_SIZE):
                        await loop.run_in_executor(None, f.write, chunk)
        # Unpack the model zip
        with tarfile.open(tmp_zipfile, "r:gz") as tar:
            tar.extractall(tmpdir)
        tmp_zipdir = tmp_zipfile.split(".")[0]
        # Check if '.pt' file exists in the extracted directory
        model_files = [f for f in os.listdir(tmp_zipdir) if f.endswith(".pt")]
        if not model_files:
            raise FileNotFoundError(
                'No ".pt" file found in the extracted directory '
                f'"{os.path.dirname(model_path)}".'
            )
        # Move over the first .pt file found to the model_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        shutil.move(os.path.join(tmp_zipdir, model_files[0]), model_path)

    return {"model_path": model_path, "model_url": model_url}


async def evaluate_model(inference_state: InferenceModelInput) -> list[str]:
    """
    Evaluate the model with the given parameters.
    Adapted from https://github.com/lantunes/CrystaLLM
    """
    torch.manual_seed(inference_state.seed)
    torch.cuda.manual_seed(inference_state.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # for later use in torch.autocast
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[inference_state.dtype]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    checkpoint = torch.load(inference_state.model_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    if inference_state.compile:
        model = torch.compile(model)

    # encode the beginning of the prompt
    prompt = inference_state.raw_input
    start_ids = encode(tokenizer.tokenize_cif(prompt))
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

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

    return generated


async def write_cif_files(result: InferenceResultsInput) -> None:
    """
    Write the generated CIFs to the specified target (console or file).
    """
    if result.generate_cif:
        for k, sample in enumerate(result.generated_samples):
            fname = f"sample_{k + 1}.cif"
            with open(fname, "wt", encoding="utf-8") as f:
                f.write(sample)
