import io
import os
from pathlib import Path
import subprocess

import pandas as pd
import tqdm

import atom3d.util.file as fi


class Scores(object):
    """
    Class for tracking and looking up Rosetta score files.

    :param file_list: List of paths to Rosetta silent files.
    :type file_list: list[Union[str, Path]]
    """

    def __init__(self, file_list):
        self._scores = {}
        file_list = [Path(x).absolute() for x in file_list]
        for silent_file in file_list:
            key = self._key_from_silent_file(silent_file)
            self._scores[key] = self._parse_scores(silent_file)

        self._scores = pd.concat(self._scores).sort_index()

    def _parse_scores(self, silent_file):
        grep_cmd = f"grep ^SCORE: {silent_file}"
        out = subprocess.Popen(
            grep_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=os.getcwd(), shell=True)
        (stdout, stderr) = out.communicate()

        f = io.StringIO(stdout.decode('utf-8'))
        return pd.read_csv(f, delimiter='\s+').drop('SCORE:', axis=1) \
            .set_index('description')

    def _key_from_silent_file(self, silent_file):
        return silent_file.stem.split('.')[0]

    def _lookup(self, file_path):
        file_path = Path(file_path)
        key = (file_path.stem, file_path.name)
        if key in self._scores.index:
            return key, self._scores.loc[key]
        key = (file_path.parent.stem, file_path.stem)
        if key in self._scores.index:
            return key, self._scores.loc[key]
        return None, None

    def __call__(self, x, error_if_missing=False):
        key, x['scores'] = self._lookup(x['file_path'])
        if key is not None:
            x['id'] = str(key)
        if x['scores'] is None and error_if_missing:
            raise RuntimeError(f'Unable to find scores for {x["file_path"]}')
        return x

    def remove_missing(self, file_list):
        """Remove examples we cannot find in score files."""
        result = []
        for i, file_path in tqdm.tqdm(enumerate(file_list), total=len(file_list)):
            entry = self._lookup(file_path)
            if entry is not None:
                result.append(file_path)
        return result

    def __len__(self):
        """Get number of individual structures across score files."""
        return len(self._scores)
