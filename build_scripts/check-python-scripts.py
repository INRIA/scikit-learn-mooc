import sys
import os

folder = sys.argv[1]
non_python_files = [fn for fn in os.listdir(folder) if not fn.endswith('.py')]

if non_python_files:
    raise RuntimeError(f'Looks like you have non python files in {folder}:\n'
                       f'{non_python_files}')
