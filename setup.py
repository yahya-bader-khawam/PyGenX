from setuptools import setup, find_packages
import codecs
import os


VERSION = '1.0'
DESCRIPTION = 'PyGenX package'

# Setting up
setup(
    name="pygenx",
    version=VERSION,
    author="SciModel.dev (Yahya Khawam)",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['langchain', 'openai']
)
