import json
import yaml
from pathlib import Path
from typing import List, Union, Dict

def load_json(path: Path) -> Dict:
    try:
        with path.open('r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"No JSON file found at the specified path: {path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON file at {path}: {str(e)}")


def write_json(data: Union[Dict, List[Dict]], path: Path) -> None:
    try:
        with path.open('w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except OSError as e:
        raise OSError(f"Failed to write JSON to {path}: {str(e)}")


def load_yaml(path: Path) -> Dict:
    try:
        with path.open('r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"No YAML file found at the specified path: {path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file at {path}: {str(e)}")
