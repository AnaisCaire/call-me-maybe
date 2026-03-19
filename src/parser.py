from pydantic import BaseModel, ValidationError
from typing import Dict, Literal, Optional, List
from pathlib import Path
import json
import os


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

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        try:
            with open(self.file_path, 'r') as f:
                def_json = json.load(f)
                for item in def_json:
                    try:
                        json_functiondef = FunctionDef(**item)
                        self.functions.append(json_functiondef)
                    except ValidationError as e:
                        print(f'[ERROR WARNING] {e}')
                        # import pdb
                        # pdb.set_trace()
                        continue
                return self.functions
        except ValueError:
            raise ValueError
