from dataclasses import dataclass


@dataclass
class InferenceInput:
    attempts: int
    model_name: str
    file: str


@dataclass
class PubChemTaskInput:
    molecule_name: str


GPU_TASK_QUEUE_NAME = "gpu-queue"
CPU_TASK_QUEUE_NAME = "cpu-queue"
