from setuptools import setup, find_packages

setup(
    name = 'mpiprocessing',
    packages = find_packages(include=['mpiprocessing']),
    install_requires = ['mpi4py', 'cloudpickle'],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],
)
