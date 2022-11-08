from __future__ import print_function
from packaging.version import Version
import sys

OK = '\x1b[42m[ OK ]\x1b[0m'
FAIL = "\x1b[41m[FAIL]\x1b[0m"

try:
    import importlib
except ImportError:
    print(FAIL, "Python version 3.6 or above is required,"
                " but %s is installed." % sys.version)


def import_version(pkg, min_ver, fail_msg=""):
    mod = None
    try:
        mod = importlib.import_module(pkg)
        if pkg in {'PIL'}:
            try:
                ver = mod.__version__
            except AttributeError:
                try:
                    ver = mod.VERSION
                except AttributeError:
                    try:
                        ver = mod.PILLOW_VERSION
                    except:
                        raise
        else:
            ver = mod.__version__
        if Version(ver) < Version(min_ver):
            print(FAIL, "%s version %s or higher required, but %s installed."
                  % (lib, min_ver, ver))
        else:
            print(OK, '%s version %s' % (pkg, ver))
    except ImportError:
        print(FAIL, '%s not installed. %s' % (pkg, fail_msg))
    return mod


# first check the python version
print('Using python in', sys.prefix)
print(sys.version)
pyversion_str = "%s.%s" % (sys.version_info.major, sys.version_info.minor)
pyversion = Version(pyversion_str)
if pyversion >= Version("3"):
    if pyversion < Version("3.6"):
        print(FAIL, "Python version 3.6 or above is required,"
                    " but %s is installed." % sys.version)
elif pyversion >= Version("2"):
    print(FAIL, "Python version 3.6 or above is required,"
                " but %s is installed." % sys.version)
else:
    print(FAIL, "Unknown Python version: %s" % sys.version)

print()
requirements = {'numpy': "1.16", 'scipy': "1.2", 'matplotlib': "3.0",
                'sklearn': "1.1", 'pandas': "1",
                'seaborn': "0.11",
                'notebook': "5.7", 'plotly': "5.10"}

# now the dependencies
for lib, required_version in list(requirements.items()):
    import_version(lib, required_version)
