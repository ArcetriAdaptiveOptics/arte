#!/usr/bin/env python
import os
import sys
from shutil import rmtree

from setuptools import setup, Command

NAME = 'arte'
DESCRIPTION = 'Arcetri Random sTuff collEction'
URL = 'https://github.com/ArcetriAdaptiveOptics/arte'
EMAIL = 'lorenzo.busoni@inaf.it'
AUTHOR = 'Lorenzo Busoni, Alfio Puglisi, INAF Arcetri Adaptive Optics group'
LICENSE = 'MIT'
KEYWORDS = 'Adaptive Optics, Astrophysics, INAF, Arcetri',

here = os.path.abspath(os.path.dirname(__file__))
# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(name=NAME,
      description=DESCRIPTION,
      version=about['__version__'],
      classifiers=['Development Status :: 4 - Beta',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3',
                   ],
      long_description=open('README.md').read(),
      url=URL,
      author_email=EMAIL,
      author=AUTHOR,
      license=LICENSE,
      keywords=KEYWORDS,
      packages=['arte',
                'arte.atmo',
                'arte.contrib',
                'arte.control',
                'arte.dataelab',
                'arte.math',
                'arte.misc',
                'arte.photometry',
                'arte.time_series',
                'arte.types',
                'arte.utils',
                ],
      package_data={
          'arte': ['data/*'],
      },
      python_requires='>=3.8.0',
      install_requires=["numpy>1.17",
                        "scipy",
                        "matplotlib",
                        "astropy",
                        "synphot",
                        "pytest",
                        "skycalc-cli",
                        "scikit-image>=0.17",
                        "scipy",
                        "mpmath",
                        "requests"
                        ],
      include_package_data=True,
      test_suite='test',
      cmdclass={'upload': UploadCommand, },
      )
