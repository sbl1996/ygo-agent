from setuptools import setup, find_packages

__version__ = "0.0.1"

INSTALL_REQUIRES = [
  "setuptools",
  "wheel",
  "pybind11-stubgen",
  "numpy",
]

setup(
    name="mcts",
    version=__version__,
    packages=find_packages(include='mcts*'),
    long_description="",
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.7",
    include_package_data=True,
)