from src.parser import FunctionDef
from typing import List, Set
from enum import Enum, auto
from pydantic import BaseModel
import numpy as np


class NumericConstraint(BaseModel):
    """
    Context-aware numeric token validator.
    Separates 'discovery' (which chars are allowed) from 'validation'
    """
    allowed_chars: Set[str] = set("0123456789.-")

    def get_valid_continuation_tokens(self, current_buffer: str,
                                      candidates: list[tuple[int, str]]
                                      ) -> List[int]:
        """
        ARG:
            - current_buffer : generated_so_far
            - candidates : pre-processed list of every token ID
                and string representation from the inverted vocabulary.
        """
        return [tid
                for tid, s in candidates
                if self._is_valid_continuation(current_buffer, s)]

    def _is_valid_continuation(self, buffer: str, token: str) -> bool:
        if not token:
            return False
        candidate = buffer + token
        if not all(c in self.allowed_chars for c in candidate):
            return False
        # Must start with a digit or '-', never '.'
        if candidate[0] not in "-0123456789":
            return False
        if "-" in candidate[1:]:
            return False
        if candidate.count(".") > 1:
            return False
        return True


class GenState(Enum):
    """Tracks where we are in the JSON structure while being generated."""
    FUNC_NAME = auto()    # generating "name"
    AFTER_NAME = auto()   # generating: ", "parameters": {"
    PARAM_KEY = auto()    # generating a parameter key (with quotes)
    AFTER_PARAM = auto()  # generating: ": "
    PARAM_VALUE = auto()  # generating a parameter value
    BETWEEN_PARAMS = auto()  # Bridge: handles ", " between parameters
    CLOSING = auto()      # writing "}}"
    DONE = auto()         # generation done


class ConstraintEngine:
    """
    Token-aware constraint engine for JSON function calls.
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
            for tid, s in vocab.items()}

        # --- SPECIALIZED FILER: for faster speed ---

        self.quote_tokens = np.array([
            tid for tid, s in self._token_clean_map.items() if s.strip() == '"'
        ], dtype=np.int32)
        self.safe_string_content = np.array([
            tid for tid, s in self._token_clean_map.items() if '"' not in s
        ], dtype=np.int32)

        # Brace Closers (Last Param -> go to CLOSING, no commas)
        # String Closers (More Params -> go to BETWEEN_PARAMS, no braces)
        safe_brace = []
        for tid, s in self._token_clean_map.items():
            if s.count('"') == 1:
                after = s.split('"')[1]
                if all(c in " }\n\r\t" for c in after):
                    safe_brace.append(tid)
        self.safe_brace_closers = np.array(safe_brace, dtype=np.int32)
        safe_comma = []
        for tid, s in self._token_clean_map.items():
            if s.count('"') == 1:
                after = s.split('"')[1]
                if all(
                    c in " ,\n\r\t" for c in after
                ) and after.count(",") <= 1:
                    safe_comma.append(tid)
        self.safe_comma_closers = np.array(safe_comma, dtype=np.int32)

        self.strict_comma_tokens = np.array([
            tid for tid, s in self._token_clean_map.items()
            if s.strip() in [",", ", "]
        ], dtype=np.int32)

        self.strict_brace_tokens = np.array([
            tid for tid, s in self._token_clean_map.items()
            if s.strip() in ["}", "}}"]
        ], dtype=np.int32)

        # Two-step architechture:
        # 1. loops 151,000 tokens with:
        # Does this token contain only numbers/decimals/minuses?
        # happends only once
        self._numeric_candidate_tokens: list[tuple[int, str]] = [
            (tid, s)
            for tid, s in self._token_clean_map.items()
            if s and all(c in "0123456789.-" for c in s)]
        # 2. checks valid number (double decimals, invalid minus sign...)
        self.numeric_constraint = NumericConstraint()

        # --- LITERAL BRIDGE INDICES ---

        self.literal_indices: dict[GenState, dict[str, np.ndarray]] = {
            GenState.AFTER_NAME:
                self._build_literal_index('", "parameters": {'),
            GenState.AFTER_PARAM:
                self._build_literal_index(': '),
            GenState.BETWEEN_PARAMS:
                self._build_literal_index(', '),
        }

        # pre-made func name
        self.func_name_index: dict[str, np.ndarray] = self._build_name_index()

        # pre-made: {func_name: {param_name: {fragment: [tid,...]}}}
        self.param_key_indices: dict[str, dict[str, dict[str, list[int]]]] = \
            self._build_param_key_indices()

        self.reset()

    # ------------------------------------------------------------------
    # the builders
    # ------------------------------------------------------------------

    def _build_literal_index(self, target: str) -> dict[str, np.ndarray]:
        """
        Pre-computes {prefix → mask} for the static literal target.
        'prefix' is how much of target has already generated.
        The mask marks every token that can legally come next.
        """
        index: dict[str, np.ndarray] = {}
        for i in range(len(target)):
            prefix = target[:i]
            remaining = target[i:]
            valid_tids = [
                tid for tid, s in self._token_clean_map.items()
                if s and remaining.startswith(s)]
            mask = np.full(self.vocab_size, -1e9, dtype=np.float32)
            if valid_tids:
                mask[valid_tids] = 0.0
            mask.flags.writeable = False
            index[prefix] = mask
        return index

    def _build_name_index(self) -> dict[str, np.ndarray]:
        """
        Pre-computes {fragment → mask} across all function names.
        fragmen = valid prefix of any function name.
        """
        all_names = [f.name for f in self.functions]
        all_prefixes: set[str] = set()
        for name in all_names:
            for i in range(len(name) + 1):
                all_prefixes.add(name[:i])

        index: dict[str, np.ndarray] = {}
        for fragment in all_prefixes:
            mask = np.full(self.vocab_size, -1e9, dtype=np.float32)
            for tid, s in self._token_clean_map.items():
                if any(n.startswith(fragment + s) for n in all_names):
                    mask[tid] = 0.0
                elif s == '"' and any(n == fragment for n in all_names):
                    mask[tid] = 0.0
            mask.flags.writeable = False
            index[fragment] = mask
        return index

    def _build_param_key_indices(
            self) -> dict[str, dict[str, dict[str, list[int]]]]:
        """
        Pre-computes per-function, per-param: {fragment → [valid_token_ids]}.
        """
        result: dict[str, dict[str, dict[str, list[int]]]] = {}
        for func in self.functions:
            result[func.name] = {}
            for param_name in func.parameters:
                param_index: dict[str, list[int]] = {}
                for i in range(len(param_name) + 1):
                    fragment = param_name[:i]
                    tids: list[int] = []
                    for tid, s in self._token_clean_map.items():
                        if s and param_name.startswith(fragment + s):
                            tids.append(tid)
                        elif s == '"' and param_name == fragment:
                            tids.append(tid)
                    param_index[fragment] = tids
                result[func.name][param_name] = param_index
        return result

    # ------------------------------------------------------------------
    #  mask generation
    # ------------------------------------------------------------------

    def get_valid_mask(self) -> np.ndarray:
        """
        Produces a logit bias mask (0.0 valid, -1e9 forbidden) for each states
        """

        if self.state == GenState.FUNC_NAME:
            fragment = self.generated_so_far.split('{"name": "')[-1]
            cached = self.func_name_index.get(fragment)
            if cached is not None:
                return cached
            raise AssertionError(
                f"FUNC_NAME error: fragment={fragment!r} "
                "is not a prefix of any known function name. Known names:",
                {[f.name for f in self.functions]})

        elif self.state == GenState.AFTER_NAME:
            already = \
                self.generated_so_far.split(self.selected_function.name)[-1]
            cached = self.literal_indices[GenState.AFTER_NAME].get(already)
            if cached is not None:
                return cached
            raise AssertionError(
                f"AFTER_NAME error: key={already!r} ",
                "not in pre-computed index. Keys present:",
                {list(self.literal_indices[GenState.AFTER_NAME])})

        elif self.state == GenState.PARAM_KEY:
            mask = np.full(self.vocab_size, -1e9, dtype=np.float32)
            if not self.in_key_body:
                mask[self.quote_tokens] = 0.0
            else:
                fragment = self.generated_so_far.split('"')[-1]
                remaining_keys = [p for p in self.selected_function.parameters
                                  if p not in self.filled_params]
                func_idx = self.param_key_indices[self.selected_function.name]
                for param in remaining_keys:
                    tids = func_idx[param].get(fragment, [])
                    if tids:
                        mask[tids] = 0.0
            return mask

        elif self.state == GenState.AFTER_PARAM:
            already = self.generated_so_far.split('"')[-1]
            cached = self.literal_indices[GenState.AFTER_PARAM].get(already)
            if cached is not None:
                return cached
            raise AssertionError(
                f"AFTER_PARAM error: key={already!r}",
                "not in pre-computed index. Keys present:",
                {list(self.literal_indices[GenState.AFTER_PARAM])})

        elif self.state == GenState.PARAM_VALUE:
            mask = np.full(self.vocab_size, -1e9, dtype=np.float32)
            param_def = self.selected_function.parameters[self.current_param]
            remaining_params = [p for p in self.selected_function.parameters
                                if p not in self.filled_params]
            more_params_follow = len(remaining_params) > 0

            if param_def.type == "string":
                if not self.in_string_value:
                    mask[self.quote_tokens] = 0.0
                else:
                    mask[self.safe_string_content] = 0.0
                    if more_params_follow:
                        mask[self.safe_comma_closers] = 0.0
                    else:
                        mask[self.safe_brace_closers] = 0.0

            elif param_def.type == "number":
                number_buffer = self.generated_so_far.split(": ")[-1]
                valid_num_ids = \
                    self.numeric_constraint.get_valid_continuation_tokens(
                        number_buffer, self._numeric_candidate_tokens)
                mask[valid_num_ids] = 0.0
                if more_params_follow:
                    mask[self.strict_comma_tokens] = 0.0
                else:
                    mask[self.strict_brace_tokens] = 0.0
            return mask

        elif self.state == GenState.BETWEEN_PARAMS:
            target = ", "
            already = ""
            for i in range(len(target), 0, -1):
                if self.generated_so_far.endswith(target[:i]):
                    already = target[:i]
                    break
            if already == target:
                mask = np.full(self.vocab_size, -1e9, dtype=np.float32)
                mask[self.quote_tokens] = 0.0
                return mask
            cached = self.literal_indices[GenState.BETWEEN_PARAMS].get(already)
            if cached is not None:
                return cached
            raise AssertionError(
                f"BETWEEN_PARAMS error: key={already!r} ",
                "not in pre-computed index. Keys present:",
                {list(self.literal_indices[GenState.BETWEEN_PARAMS])})

        elif self.state == GenState.CLOSING:
            mask = np.full(self.vocab_size, -1e9, dtype=np.float32)
            mask[self.strict_brace_tokens] = 0.0
            return mask

        return np.full(self.vocab_size, -1e9, dtype=np.float32)

    def update_state(self, token_id: int) -> None:
        """
        Navigator: Recognizes transitions based on the token picked.
        """
        token_str = self._token_clean_map.get(token_id, "")
        self.generated_so_far += token_str

        if self.state == GenState.FUNC_NAME:
            if token_str.endswith('"'):
                name = self.generated_so_far.split('{"name": "')[-1][:-1]
                self.selected_function = next(
                    f for f in self.functions if f.name == name)
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

                        # Check if a closing brace is in the quote
                        if self.state == GenState.CLOSING and "}" in token_str:
                            after_quote = token_str.split('"')[-1]
                            braces = after_quote.count("}")
                            if (
                                braces >= 2
                                or (
                                    braces == 1
                                    and self.closing_first_done
                                )
                            ):
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
        remaining = [p for p in self.selected_function.parameters
                     if p not in self.filled_params]
        if remaining:
            # Check if a comma exists in the text immediately after quote
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
