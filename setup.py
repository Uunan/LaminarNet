from setuptools import setup, find_packages

setup(
    name="laminarnet",
    version="0.6.3",
    author="Unan",
    description="A novel neural architecture merging SSM, RNN, and Hierarchical processing.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Uunan/LaminarNet",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
)
