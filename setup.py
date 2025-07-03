import setuptools
import os

version = {}
with open(os.path.join("pyenfra", "__init__.py"), "r") as vf:
    for line in vf:
        if line.strip().startswith("__version__"):
            version["__version__"] = line.split("=")[1].strip().strip('"').strip("'")
            break

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setuptools.setup(
    name="pyenfra",
    version=version.get("__version__", "0.4.0"),
    author="Krzysztof Raczynski",
    author_email="chrisr@gri.msstate.edu",
    description="A Python package for fractal and complexity analysis of time series.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chrisrac/pyenfra",
    packages=setuptools.find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.4",
        "PyWavelets>=1.1.1",
        "pyts>=0.12.0",
        "matplotlib>=3.0"
    ],
    extras_require={
        "dev": ["pytest>=6.0", "flake8", "black"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    entry_points={
    },
    include_package_data=True,
)
