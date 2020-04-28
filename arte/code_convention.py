"""
A well-behaved writer class.

It is important to help readability. Therefore, arte has code conventions.

Rules:
Line indentation is 4 spaces. No Tabs!
Maximum line length is 80 characters.
Comments should be complete sentences in English.
Modules use short, all-lowercase names with underscores (snake_case).
Class names use the UpperCamelCase convention.
Method names use all-lowercase with underscore (snake_case) convention.
Docstrings use the numpydoc convention: https://numpydoc.readthedocs.io/


Eclipse Support
The Python plugin PyDev? for Eclipse has a tool for checking source code
on conformity with PEP8 standard. Use it!
Set Pydev->Editor->Code Style->Code Formatter->Formatter style=Pydev.Formatter
Check Pydev->Editor->Code Analysis->Do Code Analysis

"""

import os


class NiceWriter:
    """
    A nice writer.

    This class is an example of code styling conventions.
    """

    A_CLASS_CONSTANT = 'whatever it is, centralize constant here'
    NUMBER_OF_SLOPES = 1600  # Not N_SLOPES. Be descriptive

    def __init__(self, path):
        self._path = path

    def public_work(self, an_argument):
        """
        One-liner description here.

        More detailed description here, using reStructuredText if needed.

        Parameters
        ----------
        an_argument: string
           argument description here.

        Returns
        -------
        k: int
           whatever.

        """
        if self._path.startswith(an_argument):
            os.chdir(self.A_CLASS_CONSTANT)
        else:
            os.remove(self._path)

        for i in range(0, 3):
            self._say_something_private(i)

        j = 5
        k = 3
        a_container = []
        while k < 9:
            a_container.append(j + k)
            k = k + 1

        return sum(a_container)

    def _say_something_private(self, x):
        """
        A private method.
        A leading underscore denotes private methods.
        """
        print(x)


class InafIsAnAcronym():
    '''
    InafIsAnAcronym is more readable than INAFIsAnAcronym
    '''
    pass
