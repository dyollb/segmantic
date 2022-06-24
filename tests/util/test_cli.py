from inspect import signature
from pathlib import Path

from segmantic.util.cli import get_default_args, is_path, validate_args


def test_get_default_args():
    def function1(file_path: Path, arg1: int, arg2: float = -1.5):
        pass

    def function2(arg1: int, file_path: Path = None):
        pass

    for fun in (function1, function2):
        sig = signature(fun)

        default_options = get_default_args(sig)
        valid_args = validate_args(default_options, sig)

        for k in sig.parameters:
            if valid_args[k] is not None:
                assert is_path(sig.parameters[k]) == isinstance(valid_args[k], Path)
