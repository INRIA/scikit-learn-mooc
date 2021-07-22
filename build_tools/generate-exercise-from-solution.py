import re
from pathlib import Path
import sys

from jupytext.myst import myst_to_notebook
import jupytext

def replace_simple_text(input_py_str):
    result = input_py_str.replace("📃 Solution for", "📝")
    return result


def remove_solution(input_py_str):
    """Removes solution from py str.

    This is based on:
    - cells having "solution" in their metadata tags when removing full cells
    - a specific comment matching "# solution" that will keep only the content
      before this comment
    """
    nb = jupytext.reads(input_py_str, fmt='py:percent')

    cell_tags_list = [c['metadata'].get('tags') for c in nb.cells]
    is_solution_list = [tags is not None and 'solution' in tags
                        for tags in cell_tags_list]
    nb.cells = [cell for cell, is_solution in zip(nb.cells, is_solution_list)
                if not is_solution]

    # now we look for custom marker comment when we want to remove partial cells
    marker = "# solution"
    pattern = re.compile(f"^{marker}.*", flags=re.MULTILINE|re.DOTALL)

    cells_to_modify = [c for c in nb.cells if c["cell_type"] == "code" and
                       marker in c["source"]]

    for c in cells_to_modify:
        c["source"] = pattern.sub("# Write your code here.", c["source"])

    py_nb_str = jupytext.writes(nb, fmt='py:percent')
    return py_nb_str
    # I seem to remember you need jupytext kernel for python files so I
    # probably don't want to remove anything here.
    # header_pattern = re.compile(r"# ---\njupytext.+# ---\s*",
    #                             re.DOTALL | re.MULTILINE)
    # return re.sub(header_pattern, "", py_nb_str)


def write_exercise_py(solution_path, exercise_path):
    input_py = solution_path.read_text()

    output_py = input_py
    for replace_func in [replace_simple_text, remove_solution]:
        output_py = replace_func(output_py)
    exercise_path.write_text(output_py)


def write_all_exercises(python_scripts_folder):
    solution_paths = Path(python_scripts_folder).glob("*_sol_*")

    for solution_path in solution_paths:
        exercise_path = Path(str(solution_path).replace("_sol_", "_ex_"))
        if not exercise_path.exists():
            print(f"{exercise_path} does not exist")

        write_exercise_py(solution_path, exercise_path)


if __name__ == "__main__":
    python_scripts_folder = Path(sys.argv[1])

    write_all_exercises(python_scripts_folder)
