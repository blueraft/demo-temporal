from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from demo_temporal.workflows.inference.activities import (
        construct_model_input,
        get_model,
        run_inference,
        write_results,
    )
    from demo_temporal.workflows.shared import InferenceInput


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput) -> None:
        await workflow.execute_activity(
            get_model,
            data,
            start_to_close_timeout=timedelta(seconds=120),
        )
        data.raw_input = await workflow.execute_activity(
            construct_model_input,
            data,
            start_to_close_timeout=timedelta(seconds=60),
        )
        data = await workflow.execute_activity(
            run_inference,
            data,
            start_to_close_timeout=timedelta(seconds=120),
        )
        await workflow.execute_activity(
            write_results,
            data,
            start_to_close_timeout=timedelta(seconds=60),
        )
        return data["generated_samples"]
