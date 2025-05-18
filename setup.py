from setuptools import setup, find_packages

setup(
    name="ssm",
    version="0.4.1",
    author="Calvin Leighton",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
)