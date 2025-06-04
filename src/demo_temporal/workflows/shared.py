from dataclasses import dataclass


@dataclass
class InferenceInput:
    """
    Input data for the inference workflow.

    Attributes:
    - attempts: Number of attempts to run the inference.
    - model_path: Path to the model file.
    - model_url: URL to download the model if not available locally.
    - input_file: Path to a file containing the input prompt.
    - raw_input: Raw input string to use as a prompt.
    - num_samples: Number of samples to draw during inference.
    - max_new_tokens: Maximum number of tokens to generate in each sample.
    - temperature: Controls the randomness of predictions (1.0 = no change, < 1.0 =
        less random, > 1.0 = more random).
    - top_k: Retain only the top_k most likely tokens, clamp others to have 0
        probability.
    - seed: Random seed for reproducibility.
    - device: Device to run the model on (e.g., 'cpu', 'cuda', 'cuda:0').
    - dtype: Data type for the model (e.g., 'float32', 'bfloat16', 'float16').
    - compile: Whether to use PyTorch 2.0 to compile the model for faster inference.
    - generate_cif: If True, the model will generate CIF files.
    - generated_samples: List to store generated samples from the model.
    """

    attempts: int
    model_path: str
    model_url: str | None = None
    input_file: str | None = None
    raw_input: str | None = None
    num_samples: int = 2
    max_new_tokens: int = 3000
    temperature: float = 0.8
    top_k: int = 10
    seed: int = 1337
    device: str = "cpu"
    dtype: str = "bfloat16"
    compile: bool = False
    generate_cif: bool = False
    generated_samples: list[str] | None = None


@dataclass
class PubChemTaskInput:
    molecule_name: str


GPU_TASK_QUEUE_NAME = "gpu-queue"
CPU_TASK_QUEUE_NAME = "cpu-queue"
