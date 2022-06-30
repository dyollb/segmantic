import json
from pathlib import Path
from typing import Any


class PathEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return f"{obj}"
        return json.JSONEncoder.default(self, obj)
