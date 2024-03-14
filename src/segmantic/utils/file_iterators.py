from os import PathLike
from pathlib import Path

from typing_extensions import Self


def find_matching_files(
    input_globs: list[Path], verbose: bool = True
) -> list[list[Path]]:
    dir_0 = Path(input_globs[0].anchor)
    glob_0 = str(input_globs[0].relative_to(dir_0))
    ext_0 = input_globs[0].name.rsplit("*")[-1]

    candidate_files = {p.name.replace(ext_0, ""): [p] for p in dir_0.glob(glob_0)}

    for other_glob in input_globs[1:]:
        dir_i = Path(other_glob.anchor)
        glob_i = str(other_glob.relative_to(dir_i))
        ext_i = other_glob.name.rsplit("*")[-1]

        for p in dir_i.glob(glob_i):
            key = p.name.replace(ext_i, "")
            if key in candidate_files:
                candidate_files[key].append(p)
            elif verbose:
                print(f"No match found for {key} : {p}")

    output_files = [v for v in candidate_files.values() if len(v) == len(input_globs)]

    if verbose:
        print(f"Number of files in {input_globs[0]}: {len(candidate_files)}")
        print(f"Number of tuples: {len(output_files)}\n")

    return output_files


class FileIterator:
    """Iterate over files in directory"""

    def __init__(
        self,
        directory: PathLike,
        glob: str = "*.nii.gz",
        skip_string: str | None = None,
    ):
        self.directory = directory
        self.glob = glob
        self.skip_string = skip_string

    def __iter__(self) -> Self:
        self.files = Path(self.directory).glob(self.glob)
        return self

    def __next__(self) -> Path:
        for file_path in self.files:
            if file_path.is_file():
                if self.skip_string is None or self.skip_string not in file_path.name:
                    return file_path
        raise StopIteration


class UniqueFileIterator:
    """Iterate over files in directory1, but not in directory2"""

    def __init__(
        self,
        directory1: PathLike,
        directory2: PathLike,
        glob1: str = "*.nii.gz",
        glob2: str = "*.nii.gz",
    ):
        self.directory1 = Path(directory1)
        self.directory2 = Path(directory2)
        self.glob1 = glob1
        self.glob2 = glob2

    def __iter__(self) -> Self:
        self.files2 = {
            file.name for file in self.directory2.glob(self.glob2) if file.is_file()
        }
        self.files1 = [
            file
            for file in self.directory1.glob(self.glob1)
            if file.is_file() and file.name not in self.files2
        ]
        self.files1_iter = iter(self.files1)
        return self

    def __next__(self) -> Path:
        return next(self.files1_iter)


class MatchingFileIterator:
    """Iterate over files contained in directory1 and directory2"""

    def __init__(
        self,
        directory1: PathLike,
        directory2: PathLike,
        glob1: str = "*.nii.gz",
    ):
        self.directory1 = Path(directory1)
        self.directory2 = Path(directory2)
        self.glob1 = glob1
        self.suffix = glob1.rsplit("*")[-1]

    def __iter__(self) -> Self:
        self.files1 = self.directory1.glob(self.glob1)
        return self

    def __next__(self) -> tuple[Path, Path]:
        while True:
            file1 = next(self.files1)
            prefix = file1.name.replace(self.suffix, "")
            for file2 in self.directory2.glob(f"*{prefix}*{self.suffix}"):
                if file1.is_file() and file2.is_file():
                    return file1, file2
