#!/usr/bin/env python
from setuptools import setup


__version__ = "$Id: setup.py 199 2017-01-14 21:27:28Z lbusoni $"

setup(name='apposto',
      description='Arcetri Python adaPtive OpticS TOols',
      version='0.1',
      classifiers=['Development Status :: 4 - Beta',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3',
                   ],
      long_description=open('README.md').read(),
      url='',
      author='lorenzo.busoni@inaf.it',
      license='MIT',
      keywords='adaptive optics',
      packages=['apposto',
                ],
      install_requires=["numpy",
                        "scipy",
                        "matplotlib",
                        ],
      include_package_data=True,
      )
