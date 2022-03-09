import json
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from makefun import wraps


def make_config(func):
    func_sig = signature(func)

    params = [
        Parameter(
            "config_file",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            default=typer.Option(None, "--config-file", "-c"),
            annotation=Optional[Path],
        ),
        Parameter(
            "print_defaults",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            default=False,
            annotation=bool,
        ),
    ]

    for p in func_sig.parameters.values():
        if p.default == Parameter.empty:
            params.append(
                Parameter(
                    p.name,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    default=typer.Option(None),
                    annotation=p.annotation,
                )
            )
        else:
            params.append(p)

    new_sig = func_sig.replace(parameters=params)

    @wraps(func, new_sig=new_sig)
    def func_wrapper(
        config_file: Path, print_defaults: bool, *args: Any, **kwargs: Any
    ) -> Callable[[Any], Any]:
        if config_file is not None:
            if print_defaults:
                default_args = {
                    k: v.default
                    if v.default is not Parameter.empty
                    else f"<required option: {v.annotation.__name__}>"
                    for k, v in func_sig.parameters.items()
                }

                def print(**kwargs):
                    config_file.write_text(json.dumps(kwargs, indent=4))

                return print(**default_args)

            cast_path = (
                lambda v, k: Path(v)
                if isinstance(func_sig.parameters[k].annotation, Path)
                else v
            )

            config: dict = json.loads(config_file.read_text())
            my_args = {k: cast_path(v, k) for k, v in config.items()}
            for k, v in kwargs.items():
                if k in my_args:
                    if v == func_sig.parameters[k].default:
                        kwargs[k] = my_args[k]
            return func(*args, **kwargs)
        return func(*args, **kwargs)

    return func_wrapper


@make_config
def some_function(foo: int, file_name: Path, bar: str = "boo") -> int:
    """Some function"""
    print(f"foo: {foo}  {type(foo)}")
    print(f"file_name: {file_name} {type(file_name)}")
    print(f"bar: {bar} {type(bar)}")
    return foo


if __name__ == "__main__":
    typer.run(some_function)
