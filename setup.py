from setuptools import find_packages, setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name='package_folder',
    version="0.0.1",
    install_requires=requirements,
    packages=find_packages()
)
