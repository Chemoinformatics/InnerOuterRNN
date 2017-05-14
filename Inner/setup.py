import setuptools
from setuptools import setup

setup(
    name='InnerApproach',
    version='0.1',
    packages=setuptools.find_packages(),
    setup_requires=['pbr>=1.9', 'setuptools>=17.1'],
    pbr=True
)
