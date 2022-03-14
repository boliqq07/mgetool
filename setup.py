#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/8/2 15:47
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

from os import path

from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(

    name='mgetool',
    version='0.0.45',
    keywords=["exports", "imports", "show", "tool", "newclass", "packbox", "draft"],
    description='This is an tool box contains tools for mgedata.'
                'Some of code are non-originality, just copy for use. All the referenced code are marked,'
                'details can be shown in their sources',
    install_requires=['pandas', 'numpy', 'sympy', 'scipy', 'joblib', 'matplotlib',
                      'seaborn', 'requests', 'tqdm', 'six'],
    include_package_data=True,
    author='wangchangxin',
    author_email='986798607@qq.com',
    python_requires='>=3.6',
    maintainer='wangchangxin',
    platforms=[
        "Windows",
        "Unix",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "test", "instance", "Instance"], ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={'console_scripts': ['mgetool = mgetool.cli.main:main', 'mt = mgetool.cli.main:main']}
)
