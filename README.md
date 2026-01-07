# arte: Arcetri Random sTuff collEction

A collection of routines and utilities
developed by the Arcetri Astrophysical Observatory's AO group.

## Documentation

Online documentation: https://arte.readthedocs.io

To build documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

See `docs/README.md` for more details.

## Installation
__arte__ is tested on python 3.6+ only

* From pip (typically old version):

```pip install arte```

* From source (latest, check that Python Package test is passing):

```
git clone https://github.com/ArcetriAdaptiveOptics/arte.git
cd arte
pip install -e .
```


![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ArcetriAdaptiveOptics/arte/pythonpackage.yml)
[![Coverage Status][codecov]][codecovlink]
[![Documentation Status](https://readthedocs.org/projects/arte/badge/?version=latest)](https://arte.readthedocs.io/en/latest/?badge=latest)
[![PyPI version][pypiversion]][pypiversionlink]

[codecov]: https://codecov.io/gh/ArcetriAdaptiveOptics/arte/branch/master/graph/badge.svg?token=ACZL30U3OM
[codecovlink]: https://codecov.io/gh/ArcetriAdaptiveOptics/arte
[pypiversion]: https://badge.fury.io/py/arte.svg
[pypiversionlink]: https://badge.fury.io/py/arte
