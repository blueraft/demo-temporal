import asyncio

from concurrent.futures import ThreadPoolExecutor
from temporalio.client import Client
from temporalio.worker import Worker

from demo_temporal.workflows.inference.activities import (
    get_model,
    construct_model_input,
    run_inference,
    write_results,
)
from demo_temporal.workflows.inference.workflow import InferenceWorkflow
from demo_temporal.workflows.shared import GPU_TASK_QUEUE_NAME


async def run_worker():
    client = await Client.connect("localhost:7233", namespace="default")

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Adjust the number of workers as needed
        worker = Worker(
            client,
            task_queue=GPU_TASK_QUEUE_NAME,
            activity_executor=executor,
            workflows=[InferenceWorkflow],
            activities=[get_model, construct_model_input, run_inference, write_results],
        )

        await worker.run()


def main():
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
