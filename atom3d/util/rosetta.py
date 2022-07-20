import io
import os
from pathlib import Path
import subprocess

import numpy as np
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
            if len(file_list) == 1:
                key = 'all'
            self._scores[key] = self._parse_scores(silent_file)

        self._scores = pd.concat(self._scores).sort_index()

    def _parse_scores(self, silent_file):
        grep_cmd = f"grep ^SCORE: '{silent_file}'"
        out = subprocess.Popen(
            grep_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=os.getcwd(), shell=True)
        (stdout, stderr) = out.communicate()

        f = io.StringIO(stdout.decode('utf-8'))
        return pd.read_csv(f, delimiter='\s+').drop('SCORE:', axis=1) \
            .set_index('description')

    def _key_from_silent_file(self, silent_file):
        tmp = silent_file.stem.split('.')[0]
        if 'farna_rebuild' in tmp:
            tmp = silent_file.parent.stem
        return tmp

    def _lookup_helper(self, key):
        # If there are multiple rows matching key, return only the first one.
        # Sometime pandas return single row pd.DataFrame, so we use .squeeze()
        # to ensure it always return a pd.Series.
        tmp = self._scores.loc[key]
        if type(tmp) == pd.DataFrame:
            tmp = tmp.head(1)

        return tmp.astype(np.float64).squeeze().to_dict()

    def _lookup(self, file_path):
        file_path = Path(file_path)
        key = (file_path.stem, file_path.name)
        if key in self._scores.index:
            return key, self._lookup_helper(key)
        key = (file_path.parent.stem, file_path.stem)
        if key in self._scores.index:
            return key, self._lookup_helper(key)
        key = (file_path.parent.parent.stem, file_path.stem)
        if key in self._scores.index:
            return key, self._lookup_helper(key)
        if len(self._scores.index.get_level_values(0).unique()):
            key = ('all', file_path.name)
            result = self._lookup_helper(key)
            key = (file_path.stem.split('_')[0],
                   '_'.join(file_path.stem.split('_')[1:]))
            return (key, result)

        return file_path.parent.stem, None

    def __call__(self, x, error_if_missing=True):
        key, x['scores'] = self._lookup(x['file_path'])
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
