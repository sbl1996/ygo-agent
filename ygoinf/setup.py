from setuptools import setup, find_packages

__version__ = "0.0.1"

INSTALL_REQUIRES = [
  "numpy",
  "optree",
  "fastapi",
  "uvicorn[standard]",
  "pydantic_settings",
  "tflite-runtime",
]

setup(
    name="ygoinf",
    version=__version__,
    packages=find_packages(include='ygoinf*'),
    long_description="",
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.10",
)