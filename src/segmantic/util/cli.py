import inspect
from pathlib import Path
from typing import Any, Dict


def is_path(param: inspect.Parameter) -> bool:
    if param.annotation != inspect.Parameter.empty and inspect.isclass(
        param.annotation
    ):
        return issubclass(param.annotation, Path)
    return False


def cast_from_path(v: Any, param: inspect.Parameter) -> Any:
    return str(v) if v and is_path(param) else v


def cast_to_path(v: Any, param: inspect.Parameter) -> Any:
    return Path(v) if v and is_path(param) else v


def get_default_args(signature: inspect.Signature) -> Dict[str, Any]:
    default_args = {
        k: cast_from_path(v.default, signature.parameters[k])
        if v.default is not inspect.Parameter.empty
        else f"<required option: {v.annotation.__name__}>"
        for k, v in signature.parameters.items()
    }
    return default_args


def validate_args(
    args: Dict[str, Any],
    signature: inspect.Signature,
) -> Dict[str, Any]:
    valid_args = {}
    for k in args:
        if k in signature.parameters:
            valid_args[k] = cast_to_path(args[k], signature.parameters[k])
        else:
            raise ValueError(f"Unexpected argument {k}")
    return valid_args


__all__ = ("get_default_args", "validate_args")
