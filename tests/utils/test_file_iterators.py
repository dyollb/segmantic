from pathlib import Path
from typing import Iterable

from segmantic.utils.file_iterators import (
    FileIterator,
    MatchingFileIterator,
    UniqueFileIterator,
)


def check_iterator_behavior(iter: Iterable):
    def ensure_tuple(val):
        return (val,) if isinstance(val, (Path, str)) else val

    for item in iter:
        for p in ensure_tuple(item):
            assert isinstance(p, Path)
            assert p.exists()


def test_FileIterator(tmp_path: Path):
    """test FileIterator"""
    file1 = tmp_path / "foo.nii.gz"
    file2 = tmp_path / "bar.nii.gz"
    file3 = tmp_path / "bar.txt"
    for f in [file1, file2, file3]:
        f.touch()

    all_files = list(FileIterator(tmp_path, glob="*"))
    assert len(all_files) == 3
    assert file1 in all_files
    assert file2 in all_files
    assert file3 in all_files

    nifti_files = list(FileIterator(tmp_path, glob="*.nii.gz"))
    assert len(nifti_files) == 2
    assert file1 in nifti_files
    assert file2 in nifti_files
    assert file3 not in nifti_files

    file_iter = FileIterator(tmp_path, glob="*.nii.gz", skip_string="bar")
    assert len([f for f in file_iter]) == 1
    check_iterator_behavior(file_iter)


def test_UniqueFileIterator(tmp_path: Path):
    """test UniqueFileIterator"""
    dir1 = tmp_path / "a"
    dir2 = tmp_path / "b"
    dir3 = tmp_path / "c"
    for d in [dir1, dir2, dir3]:
        d.mkdir(exist_ok=True, parents=True)
    file1 = dir1 / "foo.nii.gz"
    file2 = dir1 / "bar.nii.gz"
    file3 = dir2 / "bar.nii.gz"
    for f in [file1, file2, file3]:
        f.touch()

    file_iter = UniqueFileIterator(dir1, dir2)
    assert len([f for f in file_iter]) == 1
    check_iterator_behavior(file_iter)

    file_iter = UniqueFileIterator(dir1, dir3)
    assert len([f for f in file_iter]) == 2
    check_iterator_behavior(file_iter)


def test_MatchingFileIterator(tmp_path: Path):
    """test MatchingFileIterator"""
    dir1 = tmp_path / "a"
    dir2 = tmp_path / "b"
    for d in [dir1, dir2]:
        d.mkdir(exist_ok=True, parents=True)
    file1 = dir1 / "foo.nii.gz"
    file2 = dir1 / "bar.nii.gz"
    file3 = dir2 / "foo.nii.gz"
    file4 = dir2 / "bar_seg.nii.gz"
    for f in [file1, file2, file3, file4]:
        f.touch()

    file_iter = MatchingFileIterator(dir1, dir2)
    assert len([f for f in file_iter]) == 2
    check_iterator_behavior(file_iter)

    file_iter = MatchingFileIterator(dir2, dir1)
    assert len([f for f in file_iter]) == 1
