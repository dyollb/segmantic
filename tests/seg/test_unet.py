import json
from pathlib import Path

from monai.bundle import ConfigParser
from monai.transforms import Compose
from typer.testing import CliRunner

from segmantic.commands.monai_unet_cli import app

runner = CliRunner()


def test_print_defaults():

    result = runner.invoke(app, ["train-config", "-c", "foo.json", "--print-defaults"])
    assert result.exit_code == 0
    assert Path("foo.json").exists()
    Path("foo.json").unlink()


def test_load_preprocessing():
    data_dir = Path(__file__).parent.parent.resolve() / "testing_data" / "config.json"
    options = json.loads(data_dir.read_text())

    parser = ConfigParser(
        {"image_key": "image", "preprocessing": options["preprocessing"]}
    )
    parser.parse(True)
    transforms = parser.get_parsed_content("preprocessing")
    assert isinstance(transforms, Compose)


def test_load_empty_preprocessing():
    parser = ConfigParser({"image_key": "image", "preprocessing": {}})
    parser.parse(True)
    transforms = parser.get_parsed_content("preprocessing")
    assert isinstance(transforms, dict)
    assert len(transforms) == 0


def test_load_disabled_preprocessing():
    parser = ConfigParser(
        {
            "image_key": "image",
            "preprocessing": {
                "_target_": "DataStatsD",
                "_disabled_": True,
                "keys": "image",
            },
        }
    )
    parser.parse(True)
    transforms = parser.get_parsed_content("preprocessing")
    assert isinstance(transforms, type(None))


def test_load_postprocessing():
    data_dir = Path(__file__).parent.parent.resolve() / "testing_data" / "config.json"
    options = json.loads(data_dir.read_text())

    parser = ConfigParser(
        {k: options[k] for k in ("image_key", "preprocessing", "postprocessing")}
    )
    parser.parse(True)
    transforms = parser.get_parsed_content("postprocessing")
    assert isinstance(transforms, Compose)


def test_load_trainer():
    import pytorch_lightning as pl

    data_dir = Path(__file__).parent.parent.resolve() / "testing_data" / "config.json"
    options = json.loads(data_dir.read_text())

    parser = ConfigParser({"trainer": options["trainer"]})
    parser.parse(True)
    trainer = parser.get_parsed_content("trainer")
    assert isinstance(trainer, pl.Trainer)
    assert trainer.max_epochs == options["trainer"]["max_epochs"]
