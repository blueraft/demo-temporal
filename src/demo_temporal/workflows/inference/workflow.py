from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from demo_temporal.workflows.inference.activities import (
        construct_model_input,
        get_model,
        run_inference,
    )
    from demo_temporal.workflows.shared import InferenceInput


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput) -> None:
        await workflow.execute_activity(
            get_model,
            data.model_path,
            data.model_url,
            start_to_close_timeout=timedelta(seconds=120),
        )
        model_input = await workflow.execute_activity(
            construct_model_input,
            data.input,
            data.input_file,
            start_to_close_timeout=timedelta(seconds=60),
        )
        await workflow.execute_activity(
            run_inference,
            data.model_path,
            model_input,
            start_to_close_timeout=timedelta(seconds=120),
        )
