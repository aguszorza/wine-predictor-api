import os
from setuptools import setup, find_packages


def read_file(fname: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), fname), "r") as f:
        return f.read()


setup(
    name='wine_predictor_api',
    packages=find_packages(where='src', exclude=['tests']),
    package_dir={'': 'src'},
    url='https://github.com/aguszorza/wine-predictor-api',
    version=read_file("VERSION"),
    description='My Wine Predictor API Server ',
    long_description=read_file("README.md"),
    author='Beteko',
    keywords=['openapi', 'api', 'swagger'],
    python_requires=">=3.6",
    setup_requires=['wheel'],
    install_requires=read_file("requirements.txt").split("\n"),
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development  :: Application Programming Interface'
    ],
    include_package_data=True
)
