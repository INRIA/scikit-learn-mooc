import re
from pathlib import Path
import sys

from jupytext.myst import myst_to_notebook
import jupytext


def remove_solution(input_py_str):
    """Removes solution from py str.

    This is based on solution having "solution" in their cell metadata tags
    """
    nb = jupytext.reads(input_py_str, fmt='py:percent')

    cell_tags_list = [c['metadata'].get('tags') for c in nb.cells]
    is_solution_list = [tags is not None and 'solution' in tags
                        for tags in cell_tags_list]
    nb.cells = [cell for cell, is_solution in zip(nb.cells, is_solution_list)
                if not is_solution]

    py_nb_str = jupytext.writes(nb, fmt='py:percent')

    header_pattern = re.compile(r"---\njupytext.+---\s*",
                                re.DOTALL | re.MULTILINE)
    return re.sub(header_pattern, "", py_nb_str)


def write_exercise_py(input_path, output_path):
    input_py = input_path.read_text()

    output_py = remove_solution(input_py)
    output_path.write_text(output_py)


# def write_all_exercises(input_root_path, output_root_path):
#     print(input_root_path, output_root_path)
#     input_exercises = Path(input_root_path).glob("**/*quiz*.md")

#     for input_path in input_exercises:
#         # FIXME there may be a better way with the pathlib API
#         relative_path_str = re.sub(str(input_root_path) + "/?", "",
#                                    str(input_path))
#         output_path = Path(output_root_path).joinpath(relative_path_str)
#         print(str(input_path), str(output_path))
#         write_exercise_myst(input_path, output_path)

if __name__ == "__main__":
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    write_exercise_py(input_path, output_path)
