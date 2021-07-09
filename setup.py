import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="src",  # Replace with your own username
    version="0.0.1",
    author="Ngai Sum WONG",
    author_email="wongngaisum@protonmail.com",
    description="Federated Learning on non-IID data",
    url="https://github.com/wongngaisum",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
