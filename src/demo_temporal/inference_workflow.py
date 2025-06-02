from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from demo_temporal.inference_activities import run_inference
    from demo_temporal.shared import InferenceInput


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput) -> None:
        await workflow.execute_activity(
            run_inference,
            data,
            start_to_close_timeout=timedelta(seconds=120),
        )
