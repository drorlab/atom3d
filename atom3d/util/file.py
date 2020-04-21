"""File-related utilities."""
import os
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
