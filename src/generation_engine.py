from llm_sdk import Small_LLM_Model
from src.parser import FunctionDef
from src.constraint_engine import ConstraintEngine
from typing import List
import json
import numpy as np


class GenerationEngine:
    """Interacts with the LLM SDK."""
    def __init__(self,
                 functions: List[FunctionDef],
                 model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.model = Small_LLM_Model(model_name)
        self.functions = functions

        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as f:
            raw_vocab = json.load(f)
        # convert to id -> string
        inverted = {v: k for k, v in raw_vocab.items()}

        # Determine actual logit size for MASK creation
        dummy_ids = self.model.encode(" ")[0].tolist()
        model_vocab_size = len(self.model.get_logits_from_input_ids(dummy_ids))

        self.constraint_engine = ConstraintEngine(functions,
                                                  inverted,
                                                  model_vocab_size)

    def generate_call(self, prompt_text: str) -> str:
        """
        Runs the generation loop.
        1. predict (using input_ids)
        2. Constraint (apply the mask i did)
        3. Update (append the winner to the input_ids)
        4. Repeat = give back the updated list back for the next token
        """

        system_message = "You are a helpful assistant that translates natural",
        " language into JSON function calls. Use only the provided functions."
        func_json = json.dumps([f.model_dump() for f in self.functions],
                               indent=2)

        full_prompt = (
            f"<|im_start|>system\n{system_message}\n"
            f"Available functions:\n{func_json}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f'<|im_start|>assistant\n{{"name": "')

        # turn the 2d tyTorch sequence in a 1d of the "BATCH"
        input_ids = self.model.encode(full_prompt)[0].tolist()
        self.constraint_engine.reset()
        while not self.constraint_engine.is_complete():
            logits = np.array(self.model.get_logits_from_input_ids(input_ids),
                              dtype=np.float32)
            mask = self.constraint_engine.get_valid_mask()

            # Intervention via logit bias
            next_token_id = int(np.argmax(logits + mask))
            input_ids.append(next_token_id)

            # Update state machine based on token choice
            self.constraint_engine.update_state(next_token_id)

            if self.constraint_engine.is_complete():
                break
            # Safety Limit (512 or 2048 depending on prompts)
            if len(input_ids) > 512:
                break

        return self.constraint_engine.generated_so_far
