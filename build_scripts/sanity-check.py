import os
import sys
import difflib

# TODO: we could get the list from .gitignore
IGNORE_LIST = [
    '.ipynb_checkpoints',
]

folder1, folder2 = sys.argv[1:3]


def get_basename(folder):
    contents = []
    for fn in os.listdir(folder):
        content = os.path.splitext(os.path.basename(fn))[0]
        if content not in IGNORE_LIST:
            contents.append(content)
    return contents


basenames1 = sorted(get_basename(folder1))
basenames2 = sorted(get_basename(folder2))

if basenames1 != basenames2:
    only_in_folder1 = set(basenames1) - set(basenames2)
    only_in_folder2 = set(basenames2) - set(basenames1)

    raise RuntimeError(
        f'Inconsistency between folder {folder1} and {folder2}\n'
        f'Only in folder {folder1}: {only_in_folder1}\n'
        f'Only in folder {folder2}: {only_in_folder2}')
