from llm_sdk import Small_LLM_Model
from src.parser import FunctionDef
from src.constraint_engine import ConstraintEngine
from typing import List
import json
import numpy as np


class GenerationEngine:
    """Interacts with the LLM SDK."""
    def __init__(self, functions: List[FunctionDef], model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.model = Small_LLM_Model(model_name)
        self.functions = functions

        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as f:
            raw_vocab = json.load(f)
        inverted = {v: k for k, v in raw_vocab.items()}

        # Determine actual logit size (Requirement V.3.1)
        dummy_ids = self.model.encode(" ")[0].tolist()
        model_vocab_size = len(self.model.get_logits_from_input_ids(dummy_ids))

        self.constraint_engine = ConstraintEngine(functions, inverted, model_vocab_size)
        # breakpoint()

    def generate_call(self, prompt_text: str) -> str:
        """Runs the generation loop (Requirement V.3.2)."""
        system_message = "You are a helpful assistant that translates natural language into JSON function calls. Use only the provided functions."
        func_json = json.dumps([f.model_dump() for f in self.functions], indent=2)

        full_prompt = (
            f"<|im_start|>system\n{system_message}\n"
            f"Available functions:\n{func_json}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f'<|im_start|>assistant\n{{"name": "'
        )

        input_ids = self.model.encode(full_prompt)[0].tolist()
        self.constraint_engine.reset()

        while not self.constraint_engine.is_complete():
            logits = np.array(self.model.get_logits_from_input_ids(input_ids), dtype=np.float32)
            mask = self.constraint_engine.get_valid_mask()

            # Requirement V.3.3: Intervention via logit bias
            next_token_id = int(np.argmax(logits + mask))
            # breakpoint()
            input_ids.append(next_token_id)

            # Update state machine based on token choice
            self.constraint_engine.update_state(next_token_id)

            # IMMEDIATE EXIT: Chop off the generation exactly when done to prevent garbage
            if self.constraint_engine.is_complete():
                break

            # Safety Limit
            if len(input_ids) > 2048:
                break

        return self.constraint_engine.generated_so_far
