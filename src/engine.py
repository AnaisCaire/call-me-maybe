from llm_sdk import Small_LLM_Model
from src.parser import FunctionDef
from typing import List, Optional
import json
from enum import Enum, auto
import numpy as np


class GenState(Enum):
    """Tracks where we are in the JSON structure being generated."""
    FUNC_NAME = auto()    # generating the value of "name"
    AFTER_NAME = auto()   # generating the literal: ", "parameters": {"
    PARAM_KEY = auto()    # generating a parameter key (with its quotes)
    AFTER_PARAM = auto()  # generating the literal ": "
    PARAM_VALUE = auto()  # generating a parameter value
    BETWEEN_PARAMS = auto() # Bridge: handles ", " between parameters
    CLOSING = auto()      # writing the two closing braces "}}"
    DONE = auto()         # generation complete — stop


class ConstraintEngine:
    """
    Token-aware constraint engine for 100% valid JSON function calls.
    Satisfies Chapter V.3.3 (Constrained Decoding) and V.5 (Performance).
    """

    def __init__(
        self,
        functions: List[FunctionDef],
        vocab: dict,
        model_vocab_size: int,
    ) -> None:
        self.functions = functions
        self.vocab_size = model_vocab_size

        self._token_clean_map: dict[int, str] = {
            tid: s.replace("\u0120", " ")  # Normalise Ġ to space
            for tid, s in vocab.items()
        }

        # --- PRE-COMPUTED NUMPY MASKS (O(1) Speed - V.5) ---
        
        self.quote_tokens = np.array([
            tid for tid, s in self._token_clean_map.items() if s.strip() == '"'
        ], dtype=np.int32)

        # Safe String Content (NO quotes allowed inside these tokens)
        self.safe_string_content = np.array([
            tid for tid, s in self._token_clean_map.items() if '"' not in s
        ], dtype=np.int32)

        # String Closers (Last Param -> must go to CLOSING, no commas allowed)
        safe_brace = []
        for tid, s in self._token_clean_map.items():
            if s.count('"') == 1:
                after = s.split('"')[1]
                if all(c in " }\n\r\t" for c in after):
                    safe_brace.append(tid)
        self.safe_brace_closers = np.array(safe_brace, dtype=np.int32)

        # String Closers (More Params -> must go to BETWEEN_PARAMS, no braces allowed)
        safe_comma = []
        for tid, s in self._token_clean_map.items():
            if s.count('"') == 1:
                after = s.split('"')[1]
                if all(c in " ,\n\r\t" for c in after) and after.count(",") <= 1:
                    safe_comma.append(tid)
        self.safe_comma_closers = np.array(safe_comma, dtype=np.int32)

        # Strict Numbers
        self.numeric_tokens = np.array([
            tid for tid, s in self._token_clean_map.items()
            if len(s) > 0 and all(c in " 0123456789.-" for c in s)
        ], dtype=np.int32)

        self.strict_comma_tokens = np.array([
            tid for tid, s in self._token_clean_map.items() if s.strip() in [",", ", "]
        ], dtype=np.int32)

        self.strict_brace_tokens = np.array([
            tid for tid, s in self._token_clean_map.items() if s.strip() in ["}", "}}"]
        ], dtype=np.int32)

        self.reset()

    def get_valid_mask(self) -> np.ndarray:
        """Produces a logit bias mask (0.0 valid, -1e9 forbidden)."""
        mask = np.full(self.vocab_size, -1e9, dtype=np.float32)

        if self.state == GenState.FUNC_NAME:
            fragment = self.generated_so_far.split('{"name": "')[-1]
            for tid, s in self._token_clean_map.items():
                if any(f.name.startswith(fragment + s) for f in self.functions):
                    mask[tid] = 0.0
                if s == '"' and any(f.name == fragment for f in self.functions):
                    mask[tid] = 0.0

        elif self.state == GenState.AFTER_NAME:
            target = '", "parameters": {'
            already = self.generated_so_far.split(self.selected_function.name)[-1]
            remaining = target[len(already):]
            for tid, s in self._token_clean_map.items():
                if remaining.startswith(s) and len(s) > 0:
                    mask[tid] = 0.0

        elif self.state == GenState.PARAM_KEY:
            if not self.in_key_body:
                mask[self.quote_tokens] = 0.0
            else:
                fragment = self.generated_so_far.split('"')[-1]
                remaining_keys = [p for p in self.selected_function.parameters if p not in self.filled_params]
                for tid, s in self._token_clean_map.items():
                    if any(p.startswith(fragment + s) for p in remaining_keys):
                        mask[tid] = 0.0
                    if s == '"' and any(p == fragment for p in remaining_keys):
                        mask[tid] = 0.0

        elif self.state == GenState.AFTER_PARAM:
            target = ": "
            already = self.generated_so_far.split('"')[-1]
            remaining = target[len(already):]
            for tid, s in self._token_clean_map.items():
                if remaining.startswith(s) and len(s) > 0:
                    mask[tid] = 0.0

        elif self.state == GenState.PARAM_VALUE:
            param_def = self.selected_function.parameters[self.current_param]
            remaining_params = [p for p in self.selected_function.parameters if p not in self.filled_params]
            more_params_follow = len(remaining_params) > 0

            if param_def.type == "string":
                if not self.in_string_value:
                    mask[self.quote_tokens] = 0.0
                else:
                    # Apply strict Context-Aware Closers
                    mask[self.safe_string_content] = 0.0
                    if more_params_follow:
                        mask[self.safe_comma_closers] = 0.0
                    else:
                        mask[self.safe_brace_closers] = 0.0

            elif param_def.type == "number":
                mask[self.numeric_tokens] = 0.0
                if more_params_follow:
                    mask[self.strict_comma_tokens] = 0.0
                else:
                    mask[self.strict_brace_tokens] = 0.0

        elif self.state == GenState.BETWEEN_PARAMS:
            target = ", "
            already = ""
            for i in range(len(target), 0, -1):
                if self.generated_so_far.endswith(target[:i]):
                    already = target[:i]
                    break
            remaining = target[len(already):]
            
            if remaining == "": 
                mask[self.quote_tokens] = 0.0
            else:
                for tid, s in self._token_clean_map.items():
                    if remaining.startswith(s) and len(s) > 0:
                        mask[tid] = 0.0

        elif self.state == GenState.CLOSING:
            mask[self.strict_brace_tokens] = 0.0

        return mask

    def update_state(self, token_id: int) -> None:
        """Navigator: Recognizes transitions based on the token picked."""
        token_str = self._token_clean_map.get(token_id, "")
        self.generated_so_far += token_str

        if self.state == GenState.FUNC_NAME:
            if token_str.endswith('"'):
                name = self.generated_so_far.split('{"name": "')[-1][:-1]
                self.selected_function = next(f for f in self.functions if f.name == name)
                self.state = GenState.AFTER_NAME

        elif self.state == GenState.AFTER_NAME:
            if self.generated_so_far.endswith('", "parameters": {'):
                self.state = GenState.PARAM_KEY
                self.in_key_body = False

        elif self.state == GenState.PARAM_KEY:
            if token_str.endswith('"'):
                if not self.in_key_body:
                    self.in_key_body = True
                else:
                    self.in_key_body = False
                    self.current_param = self.generated_so_far.split('"')[-2]
                    self.filled_params.add(self.current_param)
                    self.state = GenState.AFTER_PARAM

        elif self.state == GenState.AFTER_PARAM:
            if self.generated_so_far.endswith(": "):
                self.state = GenState.PARAM_VALUE

        elif self.state == GenState.PARAM_VALUE:
            p_type = self.selected_function.parameters[self.current_param].type
            
            if p_type == "string":
                quote_count = token_str.count('"')
                if quote_count > 0:
                    # Toggle state if an odd number of quotes is found
                    if quote_count % 2 == 1:
                        self.in_string_value = not self.in_string_value
                    
                    # If we just closed the string, transition!
                    if not self.in_string_value:
                        self._transition_after_value()
                        
                        # BPE CATCH: Check if a closing brace sneaked in with the quote
                        if self.state == GenState.CLOSING and "}" in token_str:
                            after_quote = token_str.split('"')[-1]
                            braces = after_quote.count("}")
                            if braces >= 2 or (braces == 1 and self.closing_first_done):
                                self.state = GenState.DONE
                            elif braces == 1:
                                self.closing_first_done = True

            elif p_type == "number":
                if "," in token_str: 
                    self._transition_after_value()
                elif "}" in token_str:
                    self.closing_first_done = True
                    if token_str.count("}") >= 2:
                        self.state = GenState.DONE
                    else:
                        self.state = GenState.CLOSING

        elif self.state == GenState.BETWEEN_PARAMS:
            if self.generated_so_far.endswith(', '):
                self.state = GenState.PARAM_KEY
                self.in_key_body = False

        elif self.state == GenState.CLOSING:
            if "}" in token_str:
                count = token_str.count("}")
                if count >= 2 or self.closing_first_done:
                    self.state = GenState.DONE
                else:
                    self.closing_first_done = True

    def _transition_after_value(self) -> None:
        remaining = [p for p in self.selected_function.parameters if p not in self.filled_params]
        if remaining:
            # Check if a comma exists in the text immediately following the quote (BPE bridge detection)
            after_quote = self.generated_so_far.split('"')[-1]
            if "," in after_quote:
                self.state = GenState.PARAM_KEY
                self.in_key_body = False
            else:
                self.state = GenState.BETWEEN_PARAMS
        else:
            self.state = GenState.CLOSING

    def is_complete(self) -> bool:
        return self.state == GenState.DONE

    def reset(self) -> None:
        self.generated_so_far = '{"name": "'
        self.state = GenState.FUNC_NAME
        self.selected_function = None
        self.current_param = None
        self.filled_params = set()
        self.in_key_body = False
        self.in_string_value = False
        self.closing_first_done = False


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
