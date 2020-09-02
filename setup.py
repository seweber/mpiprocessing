from setuptools import setup, find_packages
from distutils.cmd import Command
from subprocess import check_call


class BlackCommand(Command):
    description = "format the code to conform to the PEP 8 style guide"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        check_call(["black", "-l", "120", "./"])


setup(
    name="mpiprocessing",
    packages=find_packages(include=["mpiprocessing"]),
    install_requires=["mpi4py", "cloudpickle", "numpy", "scipy"],
    setup_requires=["pytest-runner", "black"],
    tests_require=["pytest"],
    cmdclass={"format": BlackCommand,},
)
