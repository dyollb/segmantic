import json
from typing import Any

import yaml


def loads(text: str, is_json: bool) -> Any:
    if is_json:
        return json.loads(text)
    return yaml.safe_load(text)


def dumps(obj: Any, is_json: bool) -> str:
    if is_json:
        return json.dumps(obj, indent=4)
    return yaml.safe_dump(obj, stream=None, sort_keys=False)  # type: ignore [no-any-return]
