from setuptools import setup, find_packages

setup(
    name='punctuationcorr',
    version='0.1.0',
    description='A punctuation restoration model',
    packages=find_packages(),
    install_requires=[
        'transformers==4.14.0',
        'torch==1.9.1',
        'numpy==1.21.5'
    ],
)
