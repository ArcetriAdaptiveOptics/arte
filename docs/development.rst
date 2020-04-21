Development
============

Python version
--------------

* Python 3.6 or later

Builds on github automatically run all tests on Python versions
3.6, 3.7 and 3.8.

(human) Language
----------------

All functions, classes, and especially documentation shall be in English.
Use a spell checker.

Coding style
------------

- Style should follow PEP8: https://www.python.org/dev/peps/pep-0008/
- Documentation is generated automatically from docstrings, that
  should follow the numpydoc convention:  https://numpydoc.readthedocs.io/

More details at https://github.com/ArcetriAdaptiveOptics/arte/blob/master/arte/code_convention.py

Developing guide
----------------

* Register on github.com and ask to be added to the list of arte's contributors
* Install git on your computer, and configure it with your name and contact email.
* Make sure that you have Python version 3.6+ installed.

Once this has been sorted out, here are the main steps:

Clone the repository
~~~~~~~~~~~~~~~~~~~~

Clone the github repository on your computer:

* ``git clone https://github.com/ArcetriAdaptiveOptics/arte.git``
* ``cd arte``

Create a feature branch
~~~~~~~~~~~~~~~~~~~~~~~

Unless your change is trivial, it's best to create a branch
to work independently:

``git checkout -b mybranch``


Develop
~~~~~~~

Make your changes and commit them into your branch. You can make as many 
commits as you want in your branch. Make sure that each commit has a 
readable description, because once developing is complete it will appear
in the commit history for all to see, together with your name.

``git commit -m "descriptive message"``

.. note::
    These commits are only visible on your computer and not to others,
    until you merge as described later. Moreover, since they are only kept
    on your local filesystem, there is no backup unless you provide one yourself.

Make a test
~~~~~~~~~~~

Each feature should have a unit test that verifies the functionality.
Test files are in arte/test and follow the same structure as the main tree.

Execute tests with pytest and make sure that none are failing:

``pytest``

Keep your branch updated
~~~~~~~~~~~~~~~~~~~~~~~~

Due to the distributed nature of git, it is possible that updates
are made to the library while you are developing. It is recommended
that from time to time you update your branch with these changes.

* ``git checkout master`` - switch to master branch
* ``git pull`` - pull all new changes from the github repository
* ``git checkout mybranch`` - switch back to your branch
* ``git merge master`` - merge changes from trunk into your branch

If someone has modified the same files as you, a conflict will arise
at this point. You have to edit the files to a satisfactory resolution,
and commit the result.

Publish
~~~~~~~

Once the feature is complete, it's time to make it available to others.
Assuming that the previous steps are complete:

* all your changes are committed into your branch
* updates from master has been merged
* all tests pass

use this sequence to publish:

* ``git checkout master`` - switch to master branch
* ``git merge mybranch`` - merge changes from your branch into master
* ``git push`` - update master branch on github

If the branch have been kept updated, no conflicts should arise during merge.

After each push, github automatically runs all tests on a virtual machine.
To see the result, click on the small icon next to the "Latest commit..."
line on the right.
Documentation at https://arte.readthedocs.io/en/latest/index.htm is also
automatically updated.

Delete branch (optional)
~~~~~~~~~~~~~~~~~~~~~~~~

Once the new feature has been merged to master, there is no need
to keep the branch around:

* ``git branch -d mybranch``

If you plan to further develop the feature, you can keep the branch
and go on committing and merging as before.









