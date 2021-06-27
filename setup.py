import os
import re
from setuptools import setup, find_packages


# Recommendations from https://packaging.python.org/
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='atom3d',
    packages=find_packages(include=[
        'atom3d',
        'atom3d.protein',
        'atom3d.util',
        'atom3d.datasets',
        'atom3d.models',
        'atom3d.splits',
        'atom3d.filters',
        'atom3d.data',
    ]),
    version=find_version("atom3d", "__init__.py"),
    description='ATOM3D: Tasks On Molecules in 3 Dimensions',
    author='ATOM3D developers',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.atom3d.ai",
    install_requires=[
        'argparse',
        'biopython',
        'click',
        'easy-parallel',
        'freesasa',
        'h5py',
        'lmdb',
        'msgpack',
        'numpy',
        'pandas',
        'pyrr',
        'python-dotenv>=0.5.1',
        'scipy',
        'scikit-learn',
        'tables',
        'torch',
        'tqdm',
    ],
)
