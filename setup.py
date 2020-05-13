from os import path

from setuptools import setup, find_packages

setup(
    name="ma_meta_env",
    version="0.0.2",
    url="https://github.com/jzstudent/multi-env",
    py_modules=["ma_meta_env"],
    packages=find_packages(),
    author="Jiang Zhuo",
    author_email="bit.jiangz@gmail.com",
    install_requires=["matplotlib", "seaborn"],
    tests_require=["pytest"],
    python_requires=">=3.6",
)
