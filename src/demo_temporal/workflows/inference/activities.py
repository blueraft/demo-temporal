from .llm import get_model_async
from temporalio import activity

from demo_temporal.workflows.shared import InferenceInput


@activity.defn
async def get_model(model_path: str, model_url: str = None) -> dict:
    await get_model_async(model_path, model_url)


@activity.defn
async def construct_model_input(model_input: str, input_file: str) -> str:
    # also validates that the input is not empty
    if not model_input:
        with open(input_file, "r", encoding="utf-8") as f:
            model_input = f.read()
    if not model_input:
        raise ValueError("Input data cannot be empty.")

    return model_input


@activity.defn
async def run_inference(model_path, model_input) -> dict:
    # load the model and run inference
    pass
