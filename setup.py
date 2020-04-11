#!/usr/bin/env python
from setuptools import setup

setup(name='ermelab',
      description='Arcetri Random Stuff Collection',
      version='0.1',
      classifiers=['Development Status :: 4 - Beta',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   ],
      long_description=open('README.md').read(),
      url='',
      author_email='alfio.puglisi@inaf.it',
      author='Alfio Puglisi',
      license='',
      keywords='adaptive optics',
      packages=['arte'],
      install_requires=["numpy",
                        "scipy",
                        "astropy",
                        ],
      include_package_data=True,
      test_suite='test',
      )

