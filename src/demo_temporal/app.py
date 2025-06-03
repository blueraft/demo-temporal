from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from temporalio.client import Client
import uuid

from demo_temporal.workflows.inference.workflow import InferenceWorkflow
from demo_temporal.workflows.pubchem.workflow import PubChemWorkflow
from demo_temporal.workflows.shared import (
    CPU_TASK_QUEUE_NAME,
    GPU_TASK_QUEUE_NAME,
    InferenceInput,
    PubChemTaskInput,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.temporal_client = await Client.connect("localhost:7233")
        yield
    finally:
        if hasattr(app.state, "temporal_client") and app.state.temporal_client:
            await app.state.temporal_client.close()


# Instantiate FastAPI and pass the lifespan manager
app = FastAPI(lifespan=lifespan)


@app.post("/start-inference-task")
async def start_inference_task(data: InferenceInput):
    workflow_id = f"inference-workflow-{uuid.uuid4()}"
    client = app.state.temporal_client
    try:
        await client.start_workflow(
            InferenceWorkflow.run, data, id=workflow_id, task_queue=GPU_TASK_QUEUE_NAME
        )
        print("HELLO WORLD")
        return {"workflow_id": workflow_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-pubchem-task")
async def start_pubchem_task(data: PubChemTaskInput):
    client = app.state.temporal_client
    workflow_id = f"inference-workflow-{uuid.uuid4()}"
    await client.start_workflow(
        PubChemWorkflow.run, data, id=workflow_id, task_queue=CPU_TASK_QUEUE_NAME
    )
    return {"workflow_id": workflow_id}


@app.get("/get-result/{workflow_id}")
async def get_result(workflow_id: str):
    client = app.state.temporal_client
    try:
        handle = client.get_workflow_handle(workflow_id)
        result = await handle.result()
        return {"workflow_id": workflow_id, "result": result}
    except Exception as e:
        return {"workflow_id": workflow_id, "status": "not completed", "detail": str(e)}
