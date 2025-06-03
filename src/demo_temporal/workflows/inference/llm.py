import asyncio
import aiohttp
import requests
import os

BLOCK_SIZE = 1024


async def get_model_async(model_path: str, model_url: str = None) -> dict:
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


def get_model(model_path: str, model_url: str = None) -> dict:
    exists = os.path.exists(model_path)
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

    # download the model from the URL
    response = requests.get(model_url, stream=True, timeout=10)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        for data in response.iter_content(BLOCK_SIZE):
            f.write(data)

    return {"model_path": model_path, "model_url": model_url}
