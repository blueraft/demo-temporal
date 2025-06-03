import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from demo_temporal.workflows.inference.activities import run_inference
from demo_temporal.workflows.inference.workflow import InferenceWorkflow
from demo_temporal.workflows.shared import GPU_TASK_QUEUE_NAME


async def run_worker():
    client = await Client.connect("localhost:7233", namespace="default")

    worker = Worker(
        client,
        task_queue=GPU_TASK_QUEUE_NAME,
        workflows=[InferenceWorkflow],
        activities=[run_inference],
    )

    await worker.run()


def main():
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
