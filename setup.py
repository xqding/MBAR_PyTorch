from setuptools import setup, find_packages

with open("README.md", 'r') as file_handle:
    long_description = file_handle.read()

setup(
    name = "MBAR_PyTorch",
    version = "0.0.0a4",
    author = "Xinqiang (Shawn) Ding",
    author_email = "xqding@umich.edu",
    description = "A fast implementation of MBAR method using PyTorch",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/xqding/MBAR_PyTorch",
    packages = find_packages(),
    install_requires=['numpy', 'scipy', 'pytorch']
    classifiers = (
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
