from inspect import Parameter, isclass, signature
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin

import itk
import typer
from makefun import wraps

from segmantic.prepro.core import Image2, Image3


def _is_image_type(annotation: Any) -> bool:
    # special case: input is None
    if annotation is None:
        return False
    # special case: input is Union[Image2, Image3]
    if get_origin(annotation):
        if get_origin(annotation) is Union:
            return all(issubclass(t, (Image2, Image3)) for t in get_args(annotation))
    return issubclass(annotation, (Image2, Image3))


def make_cli(func):

    image_args = []

    def _translate_param(p: Parameter):
        annotation, default = p.annotation, p.default
        if _is_image_type(p.annotation):
            image_args.append(p.name)
            annotation = Path
            default = typer.Option(None) if p.default is None else typer.Option(...)
        elif p.default == Parameter.empty:
            default = typer.Option(...)
        return Parameter(
            p.name,
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
            default=default,
        )

    func_sig = signature(func)

    params = []
    last_image_argument_idx = 0
    for idx, p in enumerate(func_sig.parameters.values()):
        if _is_image_type(p.annotation):
            last_image_argument_idx = idx + 1
        elif not isclass(p.annotation):
            print(f"p.annotation = {p.annotation}")
        params.append(_translate_param(p))

    return_type = func_sig.return_annotation
    if _is_image_type(return_type):
        params.insert(
            last_image_argument_idx,
            Parameter(
                "output_file",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=typer.Option(...),
                annotation=Optional[Path],
            ),
        )
        return_type = None

    new_sig = func_sig.replace(parameters=params, return_annotation=return_type)

    @wraps(func, new_sig=new_sig)
    def func_wrapper(*args: Any, **kwargs: Any):
        output_file: Optional[Path] = None
        kwargs_inner = {}
        for k, v in kwargs.items():
            if k == "output_file":
                output_file = v
                continue
            if k in image_args and issubclass(type(v), Path):
                v = itk.imread(f"{v}")
            kwargs_inner[k] = v

        output = func(*args, **kwargs_inner)
        if output_file and output:
            return itk.imwrite(output, f"{output_file}")
        print(output)
        return output

    return func_wrapper


def register_command(app: typer.Typer, func, func_name: str = None):
    func_cli = make_cli(func)

    @app.command()
    @wraps(func_cli, func_name=func_name)
    def wrapper(*args, **kwargs):
        return func_cli(*args, **kwargs)
