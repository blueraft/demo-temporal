import asyncio
from temporalio import activity

from demo_temporal.workflows.shared import PubChemTaskInput


@activity.defn
async def run_pubchem_query(input: PubChemTaskInput) -> str:
    await asyncio.sleep(5)
    return f"molecule name: {input.molecule_name}"
