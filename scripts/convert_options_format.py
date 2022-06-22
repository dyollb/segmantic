import json
from pathlib import Path

import typer
import yaml


def main(input_file: Path, output_file: Path):
    """Convert yml to json and vice versa"""
    if input_file.suffix == "json":
        options = json.loads(input_file.read_text())
    elif input_file.suffix == "yml":
        options = yaml.safe_load(input_file.read_text())
    else:
        raise RuntimeError(f"Cannot read {input_file}. Unsupported file type.")

    if output_file.suffix == "json":
        input_file.write_text(json.dumps(options))
    elif output_file.suffix == "yml":
        input_file.write_text(yaml.safe_dump(options))
    else:
        raise RuntimeError(f"Cannot read {output_file}. Unsupported file type.")


if __name__ == "__main__":
    typer.run(main)
