import json
from inspect import signature
from pathlib import Path

import pytest
import yaml

from segmantic.utils.cli import get_default_args, is_path, validate_args


def function1(path: Path, arg_int: int, arg_float: float = -1.5):
    pass


def function2(arg_int: int, path: Path = None):
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
    args1 = {"path": "/path/file.txt", "arg_int": 10, "arg_float": 5.0}, True
    args1_skip_default = {"path": "/path/file.txt", "arg_int": 10}, True
    args1_wrong_type = {"path": 23, "arg_int": 10, "arg_float": 5}, False
    args2 = {"path": "/path/file.txt", "arg_int": 10}, True
    args2_missing_arg = {"path": "/path/file.txt"}, False
    args2_wrong_arg = {"path": "/path/file.txt", "arg_int": 10, "foo": 42}, False

    fun_args = {
        function1: (
            args1,
            args1_skip_default,
            args1_wrong_type,
        ),
        function2: (
            args2,
            args2_missing_arg,
            args2_wrong_arg,
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
