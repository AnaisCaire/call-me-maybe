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
    CLOSING = auto()      # writing the two closing braces "}}"
    DONE = auto()         # generation complete — stop


class ConstraintEngine:
    """
    Token-aware constraint engine.

    At each generation step it produces a float32 mask of shape (vocab_size,):
      0.0  → token is valid at this position
     -1e9  → token is forbidden (will be masked out by biased argmax)
    """

    def __init__(
        self,
        functions: List[FunctionDef],
        vocab: dict,           # {token_id (int): raw_token_str}
        model_vocab_size: int,
    ) -> None:
        self.functions = functions
        self.vocab_size = model_vocab_size

        # Normalise the special BPE space character once, store clean strings.
        # Used for fragment matching (function names, parameter keys).
        self._token_clean_map: dict[int, str] = {
            tid: s.replace("\u0120", " ")   # Ġ → space
            for tid, s in vocab.items()
        }

        # Pre-compute token-id sets for single characters we mask on/off.
        # Use the RAW vocab with endswith() — Qwen BPE may store punctuation
        # with a prefix byte (e.g. Ġ}, Ċ}) that survives after Ġ→space
        # normalisation only for the space character.  endswith() is the only
        # safe way to match '}', '"', and ',' regardless of prefix encoding.
        self.quote_tokens: List[int] = [
            tid for tid, s in vocab.items() if s.endswith('"')
        ]
        self.comma_tokens: List[int] = [
            tid for tid, s in vocab.items() if s.endswith(",")
        ]
        self.brace_close_tokens: List[int] = [
            tid for tid, s in vocab.items() if s.endswith("}")
        ]
        self.numeric_tokens: List[int] = [
            tid for tid, s in self._token_clean_map.items()
            if len(s) > 0 and all(c in "0123456789.-" for c in s)
        ]

        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_valid_mask(self) -> np.ndarray:
        """Return the additive logit mask for the current generation state."""
        # All tokens start forbidden; we open up only the valid ones.
        mask = np.full(self.vocab_size, -1e9, dtype=np.float32)

        if self.state == GenState.FUNC_NAME:
            # Allow any token that extends the current fragment toward a
            # known function name, plus the closing quote once the full
            # name has been written.
            fragment = self.generated_so_far.split('{"name": "')[-1]
            for tid, s in self._token_clean_map.items():
                if any(f.name.startswith(fragment + s) for f in self.functions):
                    mask[tid] = 0.0
                if s == '"' and any(f.name == fragment for f in self.functions):
                    mask[tid] = 0.0

        elif self.state == GenState.AFTER_NAME:
            # Force the exact literal: ", "parameters": {
            # Note: no trailing quote — PARAM_KEY owns the first key's opening quote.
            target = '", "parameters": {'
            already = self.generated_so_far.split(self.selected_function.name)[-1]
            remaining = target[len(already):]
            for tid, s in self._token_clean_map.items():
                if remaining.startswith(s) and len(s) > 0:
                    mask[tid] = 0.0

        elif self.state == GenState.PARAM_KEY:
            if not self.in_key_body:
                # FIX (Bug 1): entering PARAM_KEY means we need the opening
                # quote first. Force it unconditionally.
                for tid in self.quote_tokens:
                    mask[tid] = 0.0
            else:
                # Opening quote already written — allow tokens that extend
                # the fragment toward an unfilled parameter name, plus the
                # closing quote once the full name matches.
                fragment = self.generated_so_far.split('"')[-1]
                remaining_keys = [
                    p for p in self.selected_function.parameters
                    if p not in self.filled_params
                ]
                for tid, s in self._token_clean_map.items():
                    if any(p.startswith(fragment + s) for p in remaining_keys):
                        mask[tid] = 0.0
                    if s == '"' and any(p == fragment for p in remaining_keys):
                        mask[tid] = 0.0

        elif self.state == GenState.AFTER_PARAM:
            # Force the literal ": " after the closing quote of the key.
            target = ": "
            already = self.generated_so_far.split('"')[-1]
            remaining = target[len(already):]
            for tid, s in self._token_clean_map.items():
                if remaining.startswith(s) and len(s) > 0:
                    mask[tid] = 0.0

        elif self.state == GenState.PARAM_VALUE:
            param_def = self.selected_function.parameters[self.current_param]
            remaining_params = [
                p for p in self.selected_function.parameters
                if p not in self.filled_params
            ]
            more_params_follow = len(remaining_params) > 0

            if param_def.type == "string":
                if not self.in_string_value:
                    # Need the opening quote.
                    for tid in self.quote_tokens:
                        mask[tid] = 0.0
                else:
                    # Inside a string: allow any token except closing brace.
                    mask[:] = 0.0
                    for tid in self.brace_close_tokens:
                        mask[tid] = -1e9
                    # The closing quote ends the string value; it is handled
                    # in update_state. We allow it here by leaving mask[quote]=0.

            elif param_def.type == "number":
                # Allow numeric tokens to build the number.
                for tid in self.numeric_tokens:
                    mask[tid] = 0.0
                # Non-last param: terminate with ',' to move to next key.
                # Last param: terminate with '}' which closes the parameters
                # object; CLOSING then writes the outer '}'.
                if more_params_follow:
                    for tid in self.comma_tokens:
                        mask[tid] = 0.0
                else:
                    for tid in self.brace_close_tokens:
                        mask[tid] = 0.0

        elif self.state == GenState.CLOSING:
            # String params arrive here with closing_first_done=False (need inner brace).
            # Number params arrive with closing_first_done=True (inner brace already written).
            # Either way, force the next '}' unconditionally.
            for tid in self.brace_close_tokens:
                mask[tid] = 0.0

        return mask

    def update_state(self, token_id: int) -> None:
        """Advance the DFA by one generated token."""
        token_str = self._token_clean_map.get(token_id, "")
        self.generated_so_far += token_str

        if self.state == GenState.FUNC_NAME:
            if token_str.endswith('"'):
                # The name value is complete.
                name = self.generated_so_far.split('{"name": "')[-1][:-1]
                self.selected_function = next(
                    f for f in self.functions if f.name == name
                )
                self.state = GenState.AFTER_NAME

        elif self.state == GenState.AFTER_NAME:
            if self.generated_so_far.endswith('", "parameters": {'):
                self.state = GenState.PARAM_KEY
                self.in_key_body = False

        elif self.state == GenState.PARAM_KEY:
            if token_str.endswith('"'):
                if not self.in_key_body:
                    # Opening quote consumed — now inside the key body.
                    self.in_key_body = True
                else:
                    # Closing quote consumed — key is complete.
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
                if token_str.endswith('"'):
                    self.in_string_value = not self.in_string_value
                    if not self.in_string_value:
                        # Closing quote of string value seen.
                        self._transition_after_value()

            elif p_type == "number":
                if token_str.endswith(","):
                    self._transition_after_value()
                elif token_str.endswith("}"):
                    # Last number param: '}' closes the parameters object.
                    self.closing_first_done = True
                    self.state = GenState.CLOSING

        elif self.state == GenState.CLOSING:
            if token_str.endswith("}"):
                if not self.closing_first_done:
                    self.closing_first_done = True  # first brace written
                else:
                    self.state = GenState.DONE       # second brace written

    def is_complete(self) -> bool:
        """Return True when both closing braces have been written."""
        return self.state == GenState.DONE

    def reset(self) -> None:
        """Reset all mutable state for a fresh generation run."""
        self.generated_so_far: str = '{"name": "'
        self.state: GenState = GenState.FUNC_NAME
        self.selected_function: Optional[FunctionDef] = None
        self.current_param: Optional[str] = None
        self.filled_params: set = set()
        self.in_key_body: bool = False
        self.in_string_value: bool = False
        self.closing_first_done: bool = False  # tracks which of '}}' we're on

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transition_after_value(self) -> None:
        """Move to the next state after a parameter value is complete."""
        remaining = [
            p for p in self.selected_function.parameters
            if p not in self.filled_params
        ]
        if remaining:
            self.state = GenState.PARAM_KEY
            self.in_key_body = False   # opening quote not yet written
        else:
            # All parameters filled — need to close the parameters object.
            self.state = GenState.CLOSING


class GenerationEngine:
    """
    Handles prompt construction and the token-by-token generation loop.
    """

    def __init__(
        self,
        functions: List[FunctionDef],
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        self.model_name = model_name
        self.functions = functions
        self.model = Small_LLM_Model(model_name)

        # Load and invert the vocabulary: raw string → token id → clean string.
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as f:
            raw_vocab: dict = json.load(f)
        # raw_vocab is {token_str: token_id}; we need {token_id: token_str}.
        inverted: dict[int, str] = {v: k for k, v in raw_vocab.items()}

        # Probe the actual vocabulary size from one forward pass.
        dummy_ids = self.model.encode(" ")[0].tolist()
        model_vocab_size = len(self.model.get_logits_from_input_ids(dummy_ids))

        self.constraint_engine = ConstraintEngine(
            functions, inverted, model_vocab_size
        )

    def _build_prompt(
        self, user_request: str, functions: List[FunctionDef]
    ) -> str:
        """
        Assemble the ChatML prompt for Qwen3.

        The assistant turn is pre-filled with '{"name": "' so the very
        first constrained token is already the start of the function name.
        """
        func_list = [f.model_dump() for f in functions]
        functions_json = json.dumps(func_list, indent=2)
        system_message = (
            "You are a helpful assistant that translates natural language "
            "into JSON function calls. Use only the provided functions."
        )
        return (
            f"<|im_start|>system\n{system_message}\n"
            f"Available functions:\n{functions_json}<|im_end|>\n"
            f"<|im_start|>user\n{user_request}<|im_end|>\n"
            f'<|im_start|>assistant\n{{"name": "'
        )

    def generate_call(self, prompt_text: str) -> str:
        """
        Run the constrained token-by-token generation loop.

        Returns the raw generated JSON string (starting from '{"name": "').
        """
        full_prompt = self._build_prompt(prompt_text, self.functions)
        input_ids: List[int] = self.model.encode(full_prompt)[0].tolist()
        self.constraint_engine.reset()

        while not self.constraint_engine.is_complete():
            logits = np.array(
                self.model.get_logits_from_input_ids(input_ids),
                dtype=np.float32,
            )
            mask = self.constraint_engine.get_valid_mask()
            next_token_id = int(np.argmax(logits + mask))
            input_ids.append(next_token_id)
            self.constraint_engine.update_state(next_token_id)

            # Safety valve: prevent infinite loops on unexpected model behaviour.
            if len(input_ids) > 512:
                break

        return self.constraint_engine.generated_so_far