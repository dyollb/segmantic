import json
from pathlib import Path
from typing import Any

import yaml


def load(config_file: Path) -> Any:
    is_json = config_file and config_file.suffix.lower() == ".json"
    return loads(config_file.read_text(), is_json)


def loads(text: str, is_json: bool) -> Any:
    if is_json:
        return json.loads(text)
    return yaml.safe_load(text)


def dump(obj: Any, config_file: Path) -> None:
    is_json = config_file and config_file.suffix.lower() == ".json"
    config_file.write_text(dumps(obj, is_json))


def dumps(obj: Any, is_json: bool) -> str:
    if is_json:
        return json.dumps(obj, indent=4)
    return yaml.safe_dump(obj, stream=None, sort_keys=False)  # type: ignore [no-any-return]
