from llm_sdk import Small_LLM_Model
from src.parser import FunctionDef
from typing import List


class GenerationEngine:
    """
    Handles interaction with the LLM and the generation process.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.model_name = model_name


    def generate_call(self, prompt: str, functions: List[FunctionDef]) -> str:
        """
        1. Encode the prompt into Input IDs.
        2. Enter a loop to generate tokens one-by-one.
        3. Use constrained decoding (logit masking) to ensure valid JSON.
        """
        # We will build this logic step-by-step
        pass