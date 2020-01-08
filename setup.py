from setuptools import setup, find_packages

setup(
    name = 'mpimap',
    packages = find_packages(include=['mpimap']),
    install_requires = ['mpi4py', 'cloudpickle', 'numpy'],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],
)
