import argparse
from pathlib import Path
from src.parser import Parsing
from src.engine import ConstraintEngine, GenerationEngine
import numpy as np
import pdb, json


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
                        default='data/input/function_calling_results.json',
                        type=Path,
                        help='the file where we will see the output')

    arg = parser.parse_args()

    # Step 1: Load Function Definitions
    parser_instance = Parsing(arg.functions_definition)
    functions = parser_instance.load_def()

    # Step 2: Load Test Prompts (Mandatory V.2)
    with open(arg.input, 'r') as f:
        test_data = json.load(f)  # test_data is a list of {"prompt": "..."}
    generate = GenerationEngine(functions=functions)
    results = []

    # Step 3: Loop through prompts
    for entry in test_data:
        user_prompt = entry["prompt"]

        # Pass the specific prompt to the engine
        # Note: we add '{"' in the engine's build_prompt to kickstart JSON
        json_call_str = generate.generate_call(user_prompt)
        breakpoint()
        # The result must be valid JSON to be stored in the list
        try:
            # We prefix with the '{' we manually added in the prompt
            call_data = json.loads("{" + json_call_str)
            
            # Construct the final object as per V.4 requirements
            results.append({
                "prompt": user_prompt,
                "name": call_data.get("name"),
                "parameters": call_data.get("parameters")
            })
        except json.JSONDecodeError:
            print(f"Error: Model produced invalid JSON for prompt: {user_prompt}")

    # Step 4: Write the single output file (Mandatory V.4)
    with open(arg.output, 'w') as f:
        json.dump(results, f, indent=2)
    pdb.set_trace()
    breakpoint()


if __name__ == "__main__":
    main()
