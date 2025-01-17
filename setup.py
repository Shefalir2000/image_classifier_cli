from setuptools import setup, find_packages
from io import open
from os import path
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()
# automatically captured required modules for install_requires in requirements.txt and as well as configure dependency links
with open(path.join(HERE, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")
install_requires = [
    x.strip()
    for x in all_reqs
    if ("git+" not in x)
    and (not x.startswith("#"))
    and (not x.startswith("-"))
]
dependency_links = [
    x.strip().replace("git+", "") for x in all_reqs if "git+" not in x
]

setup(
    name="image_classifier_cli",
    description="Image Classifier is an easy tool that is able to help you create a new AI/ML model and predict on data using that or any model.",
    include_package_data=True,
    package_data={"": ["*.txt"]},
    version="0.0.1",
    packages=find_packages(),  # list of all packages
    install_requires=install_requires,
    python_requires=">=2.7",  # any python greater than 2.7
    entry_points="""
        [console_scripts]
        ImgClass=ImgClass.__main__:main
    """,
    author="Shefali Ranjan and Tyler Kwok",
    keywords="template, ImgClass, predict, train, easy, project, python",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    url="https://github.com/Shefalir2000/image_classifier_cli",
    download_url="",
    dependency_links=dependency_links,
    author_email="shefalir@vt.edu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
