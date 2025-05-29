from setuptools import setup, find_packages

setup(
    name="fpss",
    version="0.4.2",
    author="Calvin Leighton",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
)