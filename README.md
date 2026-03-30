*This project has been created as part of the 42 curriculum by <acaire-d>.*

# Call Me Maybe: Introduction to Function Calling in LLMs

## Description

This project implements a function calling tool that translates natural language prompts into structured JSON function calls. By utilizing constrained decoding, the system modifies the language model's token selection process at each generation step to guarantee 100% valid JSON output. This ensures near-perfect reliability even when running on a small 0.6B parameter language model like `Qwen/Qwen3-0.6B`.

## Instructions

### Installation

The project relies on `uv` for dependency management. Install the required dependencies (including `pydantic` and `numpy`) using the provided Makefile:

```bash
uv sync
```

### Execution

Run the program using the following command:

```bash
uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>]
```

By default, the program reads input files from the `data/input/` directory and writes outputs to `data/output/`.

### Available Makefile Commands

- `make install`: Install project dependencies using `uv`.
- `make run`: Execute the main script of the project.
- `make debug`: Run the main script in debug mode using Python's built-in debugger (`pdb`).
- `make lint`: Execute `flake8` and `mypy` with the required flags to ensure coding standards.
- `make clean`: Remove temporary files and caches (`__pycache__`, `.mypy_cache`) to keep the environment clean.

## Algorithm Explanation

Language models generate text token-by-token by producing a probability distribution (logits) over the entire vocabulary at each step. Constrained decoding intervenes by modifying these logits *before* token selection.

The process works as follows:

1. The model produces raw logits for all possible next tokens.
2. A DFA (Deterministic Finite Automaton) tracks the current position within the expected JSON schema.
3. Tokens that would violate the structure or schema at the current position have their logit set to `-1e9` (negative infinity).
4. The model samples exclusively from the remaining valid tokens.

The DFA transitions through the following states for each function call:

```
FUNC_NAME → AFTER_NAME → PARAM_KEY → AFTER_KEY → PARAM_VALUE
                                ↑                      |
                           PARAM_SEP ←─────────────────┘ (if more params)
                                                       |
                                                  CLOSING → DONE
```

This guarantees that every output is parseable and schema-compliant, regardless of the model's raw tendencies.

## Design Decisions

- **Strict validation:** All classes enforce validation using the `pydantic` library.
- **No magic parsing:** The target function is chosen exclusively through the LLM's constrained output — no heuristics or hardcoded logic.
- **Robust error handling:** Exceptions are handled gracefully using `try-except` blocks and context managers to ensure the program never crashes unexpectedly and always provides clear error messages.
- **Type safety:** The entire codebase uses static type hints and passes `mypy` without errors.
- **Single-char token safety:** Structural delimiters (`"`, `,`, `}`) are matched using exact equality on the cleaned token string — not `endswith()` — to prevent multi-character BPE tokens (e.g. `',"'` or `'"key"'`) from triggering the wrong DFA event. The closing brace `}` is the only exception, using `endswith("}")` to handle Qwen3's space-prefixed `}` token.

## Performance Analysis

The implementation targets the following metrics:

- **Accuracy:** 90%+ correct function selection and argument extraction.
- **Reliability:** 100% valid, schema-compliant, and parseable JSON on every output.
- **Speed:** All test prompts processed in under 5 minutes on standard hardware.

## Challenges Faced

- **BPE tokenizer quirks:** Mapping JSON schema constraints to the model's vocabulary required careful handling of subword units and special characters. Qwen3 encodes some punctuation tokens with a leading space prefix (e.g. `Ġ}` for `}`), which meant naive exact-string matching on the cleaned vocabulary silently produced empty token sets and caused the constraint engine to stall.
- **Multi-character tokens as DFA triggers:** The Qwen3 vocabulary contains compound tokens like `',"'` or `', "'` that end with a quote character. Using `endswith('"')` for quote detection allowed these tokens to fire a single DFA quote-event for what was actually two structural characters, corrupting the state machine. The fix was restricting quote and comma detection to tokens whose entire cleaned string is exactly `"` or `,`.
- **Asymmetric closing paths:** String values and number values terminate differently — strings end on a closing quote, numbers end on a `,` or `}`. This required separate closing paths that both converge correctly to the `CLOSING` state, which then writes the two `}}` braces to close the `parameters` object and the outer object.
- **Suppressing model prose:** Without aggressive masking, the 0.6B model would continue generating conversational text after the JSON was structurally complete. The DFA solves this entirely: once `DONE` is reached the generation loop exits, regardless of what the model would have produced next.

## Testing Strategy

The implementation is validated against the following cases:

- Single string parameter (e.g. `fn_greet`)
- Single number parameter (e.g. `fn_get_square_root`)
- Two number parameters (e.g. `fn_add_numbers`)
- Three string parameters (e.g. `fn_substitute_string_with_regex`)
- Mixed parameter types
- Edge cases: empty strings, large numbers, special characters, regex patterns, ambiguous prompts

Static type checking is enforced with `mypy` and coding style with `flake8`.

## Example Usage

Input prompt: `"What is the sum of 2 and 3?"`

Output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

Input prompt: `"Replace all vowels in 'hello' with asterisks"`

```json
[
  {
    "prompt": "Replace all vowels in 'hello' with asterisks",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "hello",
      "regex": "[aeiouAEIOU]",
      "replacement": "*"
    }
  }
]
```

## Resources

- [Hugging Face `transformers` documentation](https://huggingface.co/docs/transformers)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Lilian Weng — Constrained Decoding](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/#controlled-generation)

**AI Usage:** AI (Claude) was used as a Socratic tutor throughout this project — to clarify the theory behind logits and constrained decoding, to trace and diagnose bugs in the DFA logic step by step, and to format this README. No core generation or logit-masking logic was produced directly by AI; all implementation decisions were reasoned through and written by the authors.