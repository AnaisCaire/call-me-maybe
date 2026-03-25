import argparse
from pathlib import Path
from src.parser import Parsing
from src.engine import GenerationEngine
# import pdb
import sys, json, time


def main():
    """
    1. take the arguments
    2. makes sure that if none is given, there are the defaults
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

    # Step 1: Load Function Definitions
    parser_instance = Parsing(arg.functions_definition)
    functions = parser_instance.load_def()

    # Step 2: Load Test Prompts
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
    # Step 3: Loop through prompts
    start_time = time.perf_counter()
    for entry in test_data:
        user_prompt = entry["prompt"]
        generate.generate_call(user_prompt)
        print(f"DEBUG: {repr(generate.constraint_engine.generated_so_far)}")
        # The result must be valid JSON to be stored in the list
        try:
            call_data = json.loads(generate.constraint_engine.generated_so_far)
            results.append({
                "prompt": user_prompt,
                "name": call_data.get("name"),
                "parameters": call_data.get("parameters")
            })
        except json.JSONDecodeError:
            print("[ERROR]: Model produced invalid JSON for prompt:",
                  user_prompt)

    # Step 4: Write the single output file (Mandatory V.4)
    with open(arg.output, 'w') as f:
        json.dump(results, f, indent=2)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print(f"\n[SUCCESS] Results saved to {arg.output}")
    print(f"[TIMER] Total execution time: {minutes}m {seconds:.2f}s")
    # Optional: Print average time per prompt
    if len(test_data) > 0:
        avg_time = elapsed_time / len(test_data)
        print(f"[TIMER] Average time per prompt: {avg_time:.2f}s")


if __name__ == "__main__":
    main()
