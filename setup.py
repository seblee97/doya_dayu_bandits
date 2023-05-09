import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dd-bandits-seblee97",
    version="0.0.1",
    author="Sebastian Lee",
    author_email="sebastianlee.1997@yahoo.co.uk",
    description="Implementation of various bandit algorithms including doya-dayu.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seblee97/dd_bandits",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
