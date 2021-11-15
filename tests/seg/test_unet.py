from typer.testing import CliRunner
from pathlib import Path

from segmantic.commands.monai_unet_cli import app

runner = CliRunner()


def test_print_defaults():

    result = runner.invoke(app, ["train-config", "-c", "foo.json", "--print-defaults"])
    assert result.exit_code == 0
    assert Path("foo.json").exists()
    Path("foo.json").unlink()
