from dataclasses import dataclass


@dataclass
class InferenceInput:
    attempts: int
    model_path: str
    model_url: str | None = None
    input_file: str | None = None
    raw_input: str | None = None


@dataclass
class PubChemTaskInput:
    molecule_name: str


GPU_TASK_QUEUE_NAME = "gpu-queue"
CPU_TASK_QUEUE_NAME = "cpu-queue"
