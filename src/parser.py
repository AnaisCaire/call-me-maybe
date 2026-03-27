from pydantic import BaseModel, ValidationError
from typing import Dict, Literal, Optional, List
from pathlib import Path
import json
import sys


class Paramcheck(BaseModel):
    """ for the different parameters"""
    type: Literal['string', 'number']


class FunctionDef(BaseModel):
    """Validate data of functions"""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Paramcheck]
    returns: Paramcheck


class Parsing:
    """
        parse the files from JSON format to list of
        functiondef format
    """
    def __init__(self, file_path: Path) -> None:
        self.file_path: Path = file_path
        self.functions: List[FunctionDef] = []

    def load_def(self) -> List[FunctionDef]:
        if not self.file_path.exists():
            print("[ERROR] Function definition file not found:",
                  self.file_path, file=sys.stderr)
            sys.exit(1)
        try:
            with open(self.file_path, 'r') as f:
                def_json = json.load(f)
        except json.JSONDecodeError as e:
            print("[ERROR] Invalid JSON in function definition file:",
                  e, file=sys.stderr)
            sys.exit(1)

        if not isinstance(def_json, list):
            print("[ERROR] Function definitions must be a JSON array.",
                  file=sys.stderr)
            sys.exit(1)

        for item in def_json:
            try:
                json_functiondef = FunctionDef(**item)
                self.functions.append(json_functiondef)
            except ValidationError as e:
                print(f"[WARNING] Skipping invalid function schema "
                      f"'{item.get('name', 'Unknown')}': "
                      f"{e.errors()[0]['msg']}")
                continue

        if not self.functions:
            print("[ERROR] No valid functions could be loaded. Exiting.",
                  file=sys.stderr)
            sys.exit(1)

        return self.functions
