import argparse
import sys
import json
import time
# import pdb

from src.parser import Parsing, FunctionDef
from src.generation_engine import GenerationEngine

from typing import List
from pathlib import Path


def validate_call(call_data: dict, functions: List[FunctionDef]) -> str | None:
    """
    Validates final output against the loaded FunctionDef schemas.
    match functiondef to genereate call to make sure a valid call is made
    """
    name = call_data.get("name")
    func_def = next((f for f in functions if f.name == name), None)
    if func_def is None:
        return f"unknown function name: {name!r}"

    params = call_data.get("parameters")
    if not isinstance(params, dict):
        return "'parameters' is missing or not an object"

    expected = set(func_def.parameters)
    actual = set(params)

    missing = expected - actual
    if missing:
        return f"missing required parameters: {sorted(missing)}"

    extra = actual - expected
    if extra:
        return f"unexpected parameters: {sorted(extra)}"

    for param_name, param_def in func_def.parameters.items():
        value = params[param_name]
        python_type_for: dict[str, type | tuple[type, ...]] = \
            {"string": str, "number": (int, float)}
        expected_type = python_type_for[param_def.type]
        if not isinstance(value, expected_type):
            return (
                f"parameter {param_name!r} expected {param_def.type}, "
                f"got {type(value).__name__}: {value!r}"
            )

    return None


def main() -> None:
    """
    1. take the arguments
    2. makes sure that if none is given, there are the defaults
    3. parsing + loading JSON
    4. generate with constrained decoding for each prompt
    5. append result to output file
    6. timer to ensure speed
    """

    parser = argparse.ArgumentParser(description="the 3 input files")

    parser.add_argument('-f',
                        '--functions_definition',
                        default='data/input/functions_definition.json',
                        type=Path,
                        help='the file where the functions are defined')
    parser.add_argument('-i',
                        '--input',
                        default='data/input/function_calling_tests.json',
                        type=Path,
                        help='the file where we test with prompts')
    parser.add_argument('-o',
                        '--output',
                        default='data/output/function_calling_results.json',
                        type=Path,
                        help='the file where we will see the output')

    arg = parser.parse_args()

    # 1: Load Function Definitions
    parser_instance = Parsing(arg.functions_definition)
    functions = parser_instance.load_def()

    # 2: Load Test Prompts
    if not arg.input.exists():
        print(f"[ERROR] Test prompts file not found: {arg.input}",
              file=sys.stderr)
        sys.exit(1)
    try:
        with open(arg.input, 'r') as f:
            test_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in test prompts file: {e}",
              file=sys.stderr)
        sys.exit(1)

    if not isinstance(test_data, list):
        print("[ERROR] Test prompts must be a JSON array.",
              file=sys.stderr)
        sys.exit(1)

    generate = GenerationEngine(functions=functions)
    results = []
    print(f"\n[INFO] Starting generation for {len(test_data)} prompts...")
    # 3: Loop through prompts
    start_time = time.perf_counter()
    for entry in test_data:
        user_prompt = entry["prompt"]
        generate.generate_call(user_prompt)
        print(f"OUTPUT: {repr(generate.constraint_engine.generated_so_far)}")
        try:
            call_data = json.loads(generate.constraint_engine.generated_so_far)
        except json.JSONDecodeError:
            print("[ERROR] Model produced invalid JSON for prompt",
                  file=sys.stderr)
            continue

        error = validate_call(call_data, functions)
        if error:
            print("[ERROR] Schema validation failed for prompt",
                  f"{user_prompt!r}: {error}",
                  file=sys.stderr)
            continue
        results.append({
            "prompt": user_prompt,
            "name": call_data["name"],
            "parameters": call_data["parameters"],
        })

    # 4: Write the single output file
    with open(arg.output, 'w') as f:
        json.dump(results, f, indent=2)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print(f"\n[SUCCESS] Results saved to {arg.output}")
    print(f"[TIMER] Total execution time: {minutes}m {seconds:.2f}s")
    if len(test_data) > 0:
        avg_time = elapsed_time / len(test_data)
        print(f"[TIMER] Average time per prompt: {avg_time:.2f}s")


if __name__ == "__main__":
    main()
