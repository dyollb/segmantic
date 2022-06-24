import json
from inspect import signature
from pathlib import Path

import pytest
import yaml

from segmantic.util.cli import get_default_args, is_path, validate_args


def function1(file_path: Path, arg_int: int, arg_float: float = -1.5):
    pass


def function2(arg_int: int, file_path: Path = None):
    pass


def test_get_default_args():
    for fun in (function1, function2):
        sig = signature(fun)

        default_options = get_default_args(sig)
        valid_args = validate_args(default_options, sig)

        for k in sig.parameters:
            if valid_args[k] is not None:
                assert is_path(sig.parameters[k]) == isinstance(valid_args[k], Path)


def test_validate_args():
    args1 = {"file_path": "/path/file.txt", "arg_int": 10, "arg_float": 5.0}
    args1_skip_default = {"file_path": "/path/file.txt", "arg_int": 10}
    args1_wrong_type = {"file_path": 23, "arg_int": 10, "arg_float": 5}
    args2 = {"file_path": "/path/file.txt", "arg_int": 10}
    args2_missing_arg = {"file_path": "/path/file.txt"}
    args2_wrong_arg = {"file_path": "/path/file.txt", "arg_int": 10, "unknown": 42}

    fun_args = {
        function1: (
            (args1, True),
            (args1_skip_default, True),
            (args1_wrong_type, False),
        ),
        function2: (
            (args2, True),
            (args2_missing_arg, False),
            (args2_wrong_arg, False),
        ),
    }

    for fun in (function1, function2):
        sig = signature(fun)

        for args, ok in fun_args[fun]:
            if ok:
                fun(**validate_args(args, sig))
            else:
                with pytest.raises(Exception):
                    fun(**validate_args(args, sig))


def test_dump_args():
    for fun in (function1, function2):
        sig = signature(fun)

        default_options = get_default_args(sig)
        _ = json.dumps(default_options)
        _ = yaml.safe_dump(default_options)


def test_roundtrip_args():
    for fun in (function1, function2):
        sig = signature(fun)

        default_options = get_default_args(sig)

        for dump, load in ((json.dumps, json.loads), (yaml.safe_dump, yaml.safe_load)):
            args = load(dump(default_options))
            fun(**validate_args(args, sig))
