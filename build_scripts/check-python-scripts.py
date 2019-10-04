import sys
import os

# TODO: we could get the list from .gitignore
IGNORE_LIST = [
    '.ipynb_checkpoints',
]

folder = sys.argv[1]
non_python_files = [
    fn
    for fn in os.listdir(folder)
    if fn not in IGNORE_LIST and not fn.endswith('.py')
]

if non_python_files:
    raise RuntimeError(f'Looks like you have non python files in {folder}:\n'
                       f'{non_python_files}')
