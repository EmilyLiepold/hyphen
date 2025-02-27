import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Wiggles",
    version="0.0.1",
    author="Emily Liepold",
    author_email="emilyliepold@berkeley.edu",
    description="This is code which performs de-wiggling of NIRSpec IFU Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="package URL",
    project_urls={
        "Bug Tracker": "package issues URL",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy",
        "astropy",
        "matplotlib",
        "scipy",
        "stdatamodels",
        "jwst",
        "joblib",
        "dill",
    ],
)
