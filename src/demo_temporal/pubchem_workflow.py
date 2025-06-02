from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from demo_temporal.pubchem_activites import run_pubchem_query
    from demo_temporal.shared import PubChemTaskInput


@workflow.defn
class PubChemWorkflow:
    @workflow.run
    async def run(self, data: PubChemTaskInput) -> None:
        await workflow.execute_activity(
            run_pubchem_query,
            data,
            start_to_close_timeout=timedelta(seconds=10),
        )
