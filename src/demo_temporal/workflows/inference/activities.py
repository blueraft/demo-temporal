from temporalio import activity

from demo_temporal.workflows.shared import InferenceInput


@activity.defn
async def get_model(data: InferenceInput):
    from .llm import download_model

    await download_model(data.model_path, data.model_url)


@activity.defn
async def construct_model_input(data: InferenceInput) -> str:
    # also validates that the input is not empty
    if not data.raw_input and data.input_file:
        with open(data.input_file, "r", encoding="utf-8") as f:
            model_input = f.read()
            return model_input
    if not data.raw_input:
        raise ValueError("Input data cannot be empty.")
    return data.raw_input


@activity.defn
async def run_inference(data: InferenceInput) -> dict:
    from .llm import evaluate_model

    return await evaluate_model(data)


@activity.defn
async def write_results(data: InferenceInput) -> None:
    """
    Write the inference results to a file.
    """
    from .llm import write_cif_files

    await write_cif_files(data)
