from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='atom3d',
    packages=find_packages(include=[
        'atom3d',
        'atom3d.protein',
        'atom3d.shard',
        'atom3d.util',
        'atom3d.datasets',
        'atom3d.splits',
        'atom3d.filters',
        'atom3d.data',
    ]),
    version='0.1.6',
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
        'python-dotenv>=0.5.1',
        'scipy',
        'tables',
        'torch',
        'tqdm',
    ],
)
