import setuptools

__title__ = "ct2foam"
__version__ = "2.0.0"
__author__ = "Heikki Kahila"
__license__ = "GPLv3"
__copyright__ = "Copyright 2021-2026 by Heikki Kahila"


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ct2foam",
    version="2.0.0",
    author="Heikki Kahila",
    author_email="heikki.kahila@gmail.com",
    description="""A package converting Cantera chemical mechanism base thermodynamic data into OpenFoam dictionary format.""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["ct2foam", "ct2foam.tests"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "cvxopt>=1.2.0",
        "numpy>=1.20",
        "scipy>=1.5.2",
        "matplotlib>=3.3",
    ],
    entry_points={
        "console_scripts": [
            "ct2foam=ct2foam.ct2foam:main",
        ],
    },
)
