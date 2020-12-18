"""File-related utilities."""
import os
from pathlib import Path
import subprocess


def find_files(path, suffix, relative=None):
    """
    Find all files in path with given suffix. =

    :param path: Directory in which to find files.
    :type path: Union[str, Path]
    :param suffix: Suffix determining file type to search for.
    :type suffix: str
    :param relative: Flag to indicate whether to return absolute or relative path.

    :return: list of paths to all files with suffix.
    :rtype: list[Path]
    """
    if not relative:
        find_cmd = r"find {:} -regex '.*\.{:}'".format(path, suffix)
    else:
        find_cmd = r"cd {:}; find . -regex '.*\.{:}' | cut -d '/' -f 2-" \
            .format(path, suffix)
    out = subprocess.Popen(
        find_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.getcwd(), shell=True)
    (stdout, stderr) = out.communicate()
    name_list = stdout.decode().split()
    name_list.sort()
    return [Path(x) for x in name_list]


def get_pdb_code(path):
    """
    Extract 4-character PDB ID code from full path.
    
    :param path: Path to PDB file.
    :type path: str

    :return: PDB filename.
    :rtype: str
    """
    return path.split('/')[-1][:4].lower()


def get_pdb_name(path):
    """
    Extract filename for PDB file from full path.

    :param path: Path to PDB file.
    :type path: str

    :return: PDB filename.
    :rtype: str
    """
    return path.split('/')[-1]
