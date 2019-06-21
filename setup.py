"""Setup.py for DSVI package."""
from setuptools import setup

setup(
    name="dsvi",
    version="0.0.1",
    author="Daniel Greenidge",
    author_email="dev@danielgreenidge.com",
    description="Doubly stochastic variational inference for deep Gaussian processes",
    packages=["dsvi"],
    install_requires=["gpytorch", "matplotlib", "numpy", "torch"],
)
