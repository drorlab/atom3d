"""File-related utilities."""
import os
from pathlib import Path
import subprocess


def find_files(path, suffix, relative=None):
    """
    Find files in path, with given suffix.

    Optionally can specify path we want to perform the find relative to.
    """
    if not relative:
        find_cmd = "find {:} -regex '.*\.{:}'".format(path, suffix)
    else:
        find_cmd = "cd {:}; find . -regex '.*\.{:}' | cut -d '/' -f 2-" \
            .format(path, suffix)
    out = subprocess.Popen(
        find_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.getcwd(), shell=True)
    (stdout, stderr) = out.communicate()
    return stdout.decode().split()


def get_pdb_code(path_to_pdb):
    return path_to_pdb.split('/')[-1][:4].lower()


def get_pdb_name(path_to_pdb):
    return path_to_pdb.split('/')[-1]


def get_file_list(data_path, suffix):
    """
    Get list of individual file_paths from a data_path.

    A data_path can be either a file, a directory, or a list of files.

    Args:
        data_path (Union[str, Path, list[str, Path]]):
            Input to compute paths from.
        suffix (str):
            Suffix to expect for file_paths.
    """
    if isinstance(data_path, list):
        # If list, assume list of file_paths.
        return [Path(x) for x in data_path]
    else:
        data_path = Path(data_path)
        if data_path.is_dir():
            return list(data_path.glob(f'**/*{suffix}'))
        elif data_path.is_file():
            return [data_path]
        else:
            raise RuntimeError(f'Cannot read {data_path}')
            return []
