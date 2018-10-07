import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autarchy",
    version="0.0.1",
    author="Jamie Gainer",
    author_email="james.samuel.gainer@gmail.com",
    description="Fast AutoML for Structured Data: an Insight Data Science Project",
    long_description=long_description,
    url="https://github.com/JamieGainer/autarchy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['tpot>=0.9.5']
)
