import os
import sys
import difflib

folder1, folder2 = sys.argv[1:3]

basenames1 = sorted(os.path.splitext(os.path.basename(fn))[0] for fn in os.listdir(folder1))
basenames2 = sorted(os.path.splitext(os.path.basename(fn))[0] for fn in os.listdir(folder2))

if basenames1 != basenames2:
    only_in_folder1 = set(basenames1) - set(basenames2)
    only_in_folder2 = set(basenames2) - set(basenames1)

    raise RuntimeError(
        f'Inconsistency between folder {folder1} and {folder2}\n'
        f'Only in folder {folder1}: {only_in_folder1}\n'
        f'Only in folder {folder2}: {only_in_folder2}')
