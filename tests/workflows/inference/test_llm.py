import pytest
from demo_temporal.workflows.inference.llm import get_model, get_model_async


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_path, model_url, expected",
    [
        (
            "models/crystallm_v1_small/ckpt.pt",
            "https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz",
            {
                "model_path": "models/crystallm_v1_small/ckpt.pt",
                "model_url": "https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz",
            },
        ),
        ("/non/existent/path/model.pt", None, FileNotFoundError),
    ],
)
async def test_get_model_async(model_path, model_url, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            await get_model_async(model_path, model_url)
    else:
        result = await get_model_async(model_path, model_url)
        assert result == expected


@pytest.mark.parametrize(
    "model_path, model_url, expected",
    [
        (
            "models/crystallm_v1_small/ckpt.pt",
            "https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz",
            {
                "model_path": "models/crystallm_v1_small/ckpt.pt",
                "model_url": "https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz",
            },
        ),
        ("/non/existent/path/model.pt", None, FileNotFoundError),
    ],
)
def test_get_model(model_path, model_url, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            get_model(model_path, model_url)
    else:
        result = get_model(model_path, model_url)
        assert result == expected
