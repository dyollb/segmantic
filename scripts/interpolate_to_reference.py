import sitk_cli
import typer

from segmantic.image.processing import resample_to_ref

if __name__ == "__main__":
    resample_cli = sitk_cli.make_cli(resample_to_ref)
    typer.run(resample_cli)
