import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from demo_temporal.pubchem_activites import run_pubchem_query
from demo_temporal.pubchem_workflow import PubChemWorkflow
from demo_temporal.shared import CPU_TASK_QUEUE_NAME


async def run_worker():
    client = await Client.connect("localhost:7233", namespace="default")

    worker = Worker(
        client,
        task_queue=CPU_TASK_QUEUE_NAME,
        workflows=[PubChemWorkflow],
        activities=[run_pubchem_query],
    )

    await worker.run()


def main():
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
