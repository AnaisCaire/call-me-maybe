from llm_sdk import Small_LLM_Model
from src.parser import FunctionDef
from typing import List, Any, Optional, Set
import json
from enum import Enum, auto
import numpy as np, pdb


class GenState(Enum):
    """Tracks where we are in the JSON structure being generated."""
    FUNC_NAME = auto()  # generating the value of "name"
    AFTER_NAME = auto()  # generating the literal: ", "parameters": {"
    PARAM_KEY = auto()  # generating a parameter key
    PARAM_VALUE = auto()  # generating a parameter value
    DONE = auto()  # closing brace written — stop


class ConstraintEngine:
    def __init__(self,
                 functions: List[FunctionDef],
                 vocab: dict[int, str],
                 model_vocab_size: int) -> None:
        self.functions = functions
        self.vocab = vocab
        self.vocab_size = model_vocab_size
        
        # Pre-calculate categories for O(1) masking 
        self._token_clean_map = {tid: s.replace("Ġ", " ") for tid, s in vocab.items()}
        
        self.quote_tokens = [tid for tid, s in self._token_clean_map.items() if s == '"']
        self.comma_tokens = [tid for tid, s in self._token_clean_map.items() if s.strip() == ","]
        self.brace_tokens = [tid for tid, s in self._token_clean_map.items() if s.strip() == "}"]
        
        # Pre-filter numeric tokens [cite: 316]
        self.numeric_tokens = [
            tid for tid, s in self._token_clean_map.items() 
            if all(c in "0123456789.-" for c in s) and len(s) > 0
        ]

        self.reset()

    def get_valid_mask(self) -> np.ndarray:
        mask = np.zeros(self.vocab_size, dtype=np.float32)

        if self.state == GenState.FUNC_NAME:
            fragment = self.generated_so_far.split('{"name": "')[-1]
            # Optimization: Only loop through known functions, not full vocab
            for f in self.functions:
                if f.name.startswith(fragment):
                    # We need the token that matches the next part of f.name
                    remaining = f.name[len(fragment):]
                    # This still requires some lookup, but only for specific tokens
                    # A better way is to pre-map which tokens start with which strings.
            # Simplified for speed:
            for tid, s in self._token_clean_map.items():
                if any(f.name.startswith(fragment + s) for f in self.functions):
                    mask[tid] = 1.0
                if s == '"' and any(f.name == fragment for f in self.functions):
                    mask[tid] = 1.0

        elif self.state == GenState.AFTER_NAME:
            target = '", "parameters": {"'
            suffix = self.generated_so_far.split(self.selected_function.name)[-1]
            remaining = target[len(suffix):]
            for tid, s in self._token_clean_map.items():
                if remaining.startswith(s) and len(s) > 0:
                    mask[tid] = 1.0

        elif self.state == GenState.PARAM_KEY:
            # Logic similar to FUNC_NAME but for keys [cite: 300]
            last_part = self.generated_so_far.split('{')[-1].split(',')[-1].strip()
            fragment = last_part.replace('"', '') if '"' in last_part else ""
            
            for tid, s in self._token_clean_map.items():
                if any(p.startswith(fragment + s) for p in self.selected_function.parameters 
                       if p not in self.filled_params):
                    mask[tid] = 1.0
                if s == '"' and any(p == fragment for p in self.selected_function.parameters 
                                   if p not in self.filled_params):
                    mask[tid] = 1.0

        elif self.state == GenState.PARAM_VALUE:
            param_def = self.selected_function.parameters[self.current_param]
            if param_def.type == 'string':
                if not self.in_string_value:
                    mask[self.quote_tokens] = 1.0
                else:
                    mask[:] = 1.0 # Allow all inside string, logic in update_state closes it
            elif param_def.type == 'number':
                mask[self.numeric_tokens] = 1.0
                mask[self.comma_tokens] = 1.0
                mask[self.brace_tokens] = 1.0

        return mask

    def update_state(self, token_id: int) -> None:
        token_str = self._token_clean_map.get(token_id, "")
        self.generated_so_far += token_str
        
        # State transitions [cite: 271, 284]
        if self.state == GenState.FUNC_NAME and token_str == '"':
            name = self.generated_so_far.split('{"name": "')[-1][:-1]
            self.selected_function = next(f for f in self.functions if f.name == name)
            self.state = GenState.AFTER_NAME
            
        elif self.state == GenState.AFTER_NAME:
            if self.generated_so_far.endswith('", "parameters": {"'):
                self.state = GenState.PARAM_KEY
                
        elif self.state == GenState.PARAM_KEY and token_str == '"':
            # Extract key between last { or , and current "
            parts = self.generated_so_far.replace(' ', '').split('"')
            self.current_param = parts[-2]
            self.filled_params.add(self.current_param)
            self.state = GenState.PARAM_VALUE
            
        elif self.state == GenState.PARAM_VALUE:
            p_type = self.selected_function.parameters[self.current_param].type
            if p_type == 'string':
                if token_str == '"':
                    self.in_string_value = not self.in_string_value
                    if not self.in_string_value: # Just closed the string
                        self._transition_from_value()
            elif p_type == 'number':
                if token_str in [',', '}', ', ', '} ']:
                    self._transition_from_value()

    def _transition_from_value(self) -> None:
        if len(self.filled_params) == len(self.selected_function.parameters):
            self.state = GenState.DONE
        else:
            self.state = GenState.PARAM_KEY

    def is_complete(self) -> bool:
        return self.state == GenState.DONE

    def reset(self) -> None:
        self.generated_so_far = '{"name": "'
        self.state = GenState.FUNC_NAME
        self.selected_function = None
        self.current_param = None
        self.filled_params = set()
        self.in_string_value = False


class GenerationEngine:
    """
    Handles interaction with the LLM and the generation process.
    """
    def __init__(self, functions: List[FunctionDef], model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.model_name = model_name
        self.functions = functions
        self.model = Small_LLM_Model(model_name)
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, 'r') as f:
            original_dict = json.load(f)
        invert_vocab = {value: key for key, value in original_dict.items()}
        # --- NEW: Define model_vocab_size by probing the SDK  ---
        # 1. Create a tiny dummy sequence
        dummy_input = self.model.encode(" ") # Returns Tensor 
        dummy_ids = dummy_input[0].tolist() 

        # 2. Get one set of logits from the model 
        dummy_logits = self.model.get_logits_from_input_ids(dummy_ids)
        # 3. THIS is your defined size (e.g., 151936)
        model_vocab_size = len(dummy_logits)
        self.constraint_engine = ConstraintEngine(functions, invert_vocab, model_vocab_size)

    def _build_prompt(self, user_request: str, functions: List[FunctionDef]) -> str:
        """
        Assembles the ChatML prompt to guide the LLM toward function selection.
        References: Chapter V.3.2 (The Generation Pipeline).
        """
        # Convert Pydantic objects to a clean list of dicts for the prompt
        func_list = [f.model_dump() for f in functions]
        functions_json = json.dumps(func_list, indent=2)

        # System Instructions: Clear, concise, and professional
        system_message = (
            "You are a helpful assistant that translates natural language into "
            "JSON function calls. Use only the provided functions."
        )

        # Assemble ChatML (Mandatory format for Qwen/Qwen3)
        # We append '{"name": "' to force the model to follow the required schema 
        # from the very first character of the assistant response.
        prompt = (
            f"<|im_start|>system\n{system_message}\n"
            f"Available functions:\n{functions_json}<|im_end|>\n"
            f"<|im_start|>user\n{user_request}<|im_end|>\n"
            f"<|im_start|>assistant\n{{\"name\": \""
        )
        return prompt

    def generate_call(self, prompt_text: str) -> str:
        """
        Implements the generation pipeline described in V.3.2.
        """
        full_prompt_string = self._build_prompt(prompt_text, self.functions)
        # 1. SDK encode returns a Tensor (usually shape [1, seq_len])
        input_ids_tensor = self.model.encode(full_prompt_string)

        # Convert to a flat Python list for easy appending (logic outside SDK)
        # The SDK's encode gives a 2D tensor, so we take the first row
        current_sequence = input_ids_tensor[0].tolist()
        generated_ids: List[int] = []
        self.constraint_engine.reset()

        while not self.constraint_engine.is_complete():
            # V.3.1: get_logits_from_input_ids requires a list of ints
            logits_list = self.model.get_logits_from_input_ids(current_sequence)
            logits_array = np.array(logits_list, dtype=np.float32)
            
            # Get your validity mask (0.0 or 1.0)
            mask = self.constraint_engine.get_valid_mask()
            
            # Apply the bias (Negative Infinity for invalid tokens)
            # Requirement V.3.3: constrained decoding interventions
            biased_logits = logits_array + (mask - 1.0) * 1e9
            
            # Greedy selection (Chapter V.3.2 step 6)
            next_token_id = int(np.argmax(biased_logits))
            
            # Update sequence for next LLM pass and update the DFA state
            current_sequence.append(next_token_id)
            generated_ids.append(next_token_id)
            self.constraint_engine.update_state(next_token_id)
            
            # Safety break to avoid infinite loops (optional but recommended)
            if len(generated_ids) > 512:
                break
            
        return self.model.decode(generated_ids)

