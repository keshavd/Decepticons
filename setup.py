import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="decepticons",
    version="0.0.1",
    author="Keshav Dial",
    author_email="keshav@magarveylab.ca",
    description="An Extension of transformers",
    install_requires=[
        'torch',
        'pytorch-crf',
        'transformers',
        'tokenizers'
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keshavd/decepticons",
    project_urls={
        "Bug Tracker": "https://github.com/keshavd/decepticons/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
