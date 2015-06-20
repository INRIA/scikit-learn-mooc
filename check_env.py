from distutils.version import LooseVersion as Version
import sys
import numpy
import matplotlib
import scipy
import IPython
import sklearn

# first check the python version
pyversion = Version(sys.version)
if pyversion >= "3":
    if pyversion < "3.4":
        print("ERROR: Python version 3.4 (or 2.7) is required, but %s is installed." % sys.version)
elif pyversion >= "2":
    if pyversion < "2.7":
        print("ERROR: Python version 2.7 is required, but %s is installed." % sys.version)
else:
    print("ERROR: Unknown Python version: %s" % sys.version)


requirements = {numpy: "1.6.1", scipy: "0.9", matplotlib: "1.0", IPython: "3.0",
                sklearn: "0.15"}

for lib, required_version in requirements.items():
    if Version(lib.__version__) < required_version:
        print("ERROR: %s version %s or higher required, but %s installed."
              % (lib.__name__, required_version, lib.__version__))
