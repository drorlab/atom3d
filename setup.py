from setuptools import find_packages, setup

setup(
    name='atom3d',
    packages=find_packages(),
    version='0.1.0',
    description='ATOM3D: Tasks On Molecules in 3 Dimensions',
    author='Raphael Townshend',
    license='MIT',
    install_requires=[
        'argparse',
        'biopython',
        'click',
        'easy-parallel',
        'h5py',
        'numpy',
        'pandas',
        'python-dotenv>=0.5.1',
        'scipy',
        'tables',
        'tqdm',
    ],
)
