from dataclasses import dataclass


@dataclass
class InferenceInput:
    attempts: int
    model_path: str
    model_url: str = None
    input_file: str = None
    input: str = None


@dataclass
class PubChemTaskInput:
    molecule_name: str


GPU_TASK_QUEUE_NAME = "gpu-queue"
CPU_TASK_QUEUE_NAME = "cpu-queue"
