from distutils.version import LooseVersion as Version
import numpy
import matplotlib
import scipy
import IPython
import sklearn

requirements = {numpy: "1.6.1", scipy: "0.9", matplotlib: "1.0", IPython: "3.0",
                sklearn: "0.15"}

for lib, required_version in requirements.items():
    if Version(lib.__version__) < required_version:
        print("ERROR: %s version %s or higher required, but %s installed."
              % (lib.__name__, required_version, lib.__version__))
