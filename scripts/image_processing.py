import typer

from segmantic.commands.image_command import register_command
from segmantic.prepro.modality import bias_correct, scale_clamp_ct

app = typer.Typer()

register_command(app, bias_correct)
register_command(app, scale_clamp_ct)

if __name__ == "__main__":
    app()
