# Installation and Updating

The most reliable way to install `kelp-o-matic` is with [Conda](https://docs.anaconda.com/anaconda/).

The library is currently available for Python versions 3.7 through 3.10. Support for future versions 
will be added when possible.

New versions of the tool are occasionally released to improve segmentation performance, speed, and
the user interface of the tool. Changes are published to the PyPI and Anaconda repositories using
[semantic versioning](https://semver.org/). You may want to occasionally run the update commands to ensure
that you're using the most up-to-date version of `kelp-o-matic`.

## With Anaconda

### Install 

Use the Anaconda Navigator GUI to create a new environment and add the *conda-forge*, and *pytorch* channels
before searching for and installing the `kelp-o-matic` package in your environment.

Alternatively, install using your terminal or the Anaconda prompt (for Windows users) by running the following command:

```bash
conda install -c pytorch -c conda-forge kelp-o-matic
```

### Update

You can update the package when new versions become available with:

```bash
conda update -c pytorch -c conda-forge kelp-o-matic
```

## With PIP

!!! warning
    It is highly recommended to install the library with Conda, not with PIP.

You can install `kelp-o-matic` with PIP. Automatic hardware acceleration is only supported with the Conda install.

### Install

```bash
pip install kelp-o-matic
```

### Update

```bash
pip install --upgrade kelp-o-matic
```