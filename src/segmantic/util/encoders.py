import json
from pathlib import Path


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return f"{obj}"
        return json.JSONEncoder.default(self, obj)
