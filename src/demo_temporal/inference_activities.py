import asyncio
from temporalio import activity

from demo_temporal.shared import InferenceInput


@activity.defn
async def run_inference(inference_input: InferenceInput) -> str:
    await asyncio.sleep(60)
    return f"{inference_input.model_name} - {inference_input.file}"
