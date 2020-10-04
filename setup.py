import setuptools

version = "0.1"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roufcp", # Replace with your own username
    version=version,
    author="Subhrajyoty Roy",
    author_email="subhrajyotyroy@gmail.com",
    description="A python package for detecting gradual changepoint using Fuzzy Rough CP (roufCP)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/subroy13/roufcp",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'statsmodels'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.8',
    test_suite='nose.collector',
    tests_require=['nose', 'numpy'],
    license = 'MIT'
)
