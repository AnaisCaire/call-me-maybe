import argparse
from pathlib import Path
from src.parser import Parsing


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
    parser_instance = Parsing(arg.functions_definition)
    load_json_test = parser_instance.load_def()
    # to verify parser instance run debug with 'p parser_instance.file_path'
    breakpoint()


if __name__ == "__main__":
    main()
