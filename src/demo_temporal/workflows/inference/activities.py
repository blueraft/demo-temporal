import asyncio
from temporalio import activity

from demo_temporal.workflows.shared import InferenceInput


@activity.defn
async def run_inference(inference_input: InferenceInput) -> str:
    import numpy as np

    result = np.add(1, 4)
    await asyncio.sleep(60)
    return f"{inference_input.model_name} - {inference_input.file} - {result}"
