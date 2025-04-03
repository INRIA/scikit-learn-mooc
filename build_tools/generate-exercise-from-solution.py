import re
from pathlib import Path
import sys

from jupytext.myst import myst_to_notebook
import jupytext


WRITE_YOUR_CODE_COMMENT = "# Write your code here."


def replace_simple_text(input_py_str):
    result = input_py_str.replace("📃 Solution for", "📝")
    return result


def remove_solution(input_py_str):
    """Removes solution from python scripts content str.

    This is based on:
    - cells having "solution" in their metadata tags when removing full cells
    - a specific comment matching "# solution" that will keep only the content
      before this comment and add "# Write your code here." at the end of the
      cell.
    """
    nb = jupytext.reads(input_py_str, fmt="py:percent")

    cell_tags_list = [c["metadata"].get("tags") for c in nb.cells]
    is_solution_list = [
        tags is not None and "solution" in tags for tags in cell_tags_list
    ]
    # Completely remove cells with "solution" tags
    nb.cells = [
        cell
        for cell, is_solution in zip(nb.cells, is_solution_list)
        if not is_solution
    ]

    # Partial cell removal based on "# solution" comment
    marker = "# solution"
    pattern = re.compile(f"^{marker}.*", flags=re.MULTILINE | re.DOTALL)

    cells_to_modify = [
        c
        for c in nb.cells
        if c["cell_type"] == "code" and marker in c["source"]
    ]

    for c in cells_to_modify:
        c["source"] = pattern.sub(WRITE_YOUR_CODE_COMMENT, c["source"])

    previous_cell_is_write_your_code = False
    all_cells_before_deduplication = nb.cells
    nb.cells = []
    for c in all_cells_before_deduplication:
        if c["cell_type"] == "code" and c["source"] == WRITE_YOUR_CODE_COMMENT:
            current_cell_is_write_your_code = True
        else:
            current_cell_is_write_your_code = False
        if (
            current_cell_is_write_your_code
            and previous_cell_is_write_your_code
        ):
            # Drop duplicated "write your code here" cells.
            continue
        nb.cells.append(c)
        previous_cell_is_write_your_code = current_cell_is_write_your_code

    # TODO: we could potentially try to avoid changing the input file jupytext
    # header since this info is rarely useful. Let's keep it simple for now.
    py_nb_str = jupytext.writes(nb, fmt="py:percent")
    return py_nb_str


def write_exercise(solution_path, exercise_path):
    print(f"Writing exercise to {exercise_path} from solution {solution_path}")
    input_str = solution_path.read_text()

    output_str = input_str
    for replace_func in [replace_simple_text, remove_solution]:
        output_str = replace_func(output_str)
    exercise_path.write_text(output_str)


def write_all_exercises(python_scripts_folder):
    solution_paths = Path(python_scripts_folder).glob("*_sol_*")

    for solution_path in solution_paths:
        exercise_path = Path(str(solution_path).replace("_sol_", "_ex_"))
        if not exercise_path.exists():
            print(
                f"{exercise_path} does not exist, generating it from solution."
            )

        write_exercise(solution_path, exercise_path)


if __name__ == "__main__":
    path = Path(sys.argv[1])

    if path.is_dir():
        write_all_exercises(path)
    else:
        if "_ex_" not in str(path):
            raise ValueError(
                f"Path argument should be an exercise file. Path was {path}"
            )
        solution_path = Path(str(path).replace("_ex_", "_sol_"))
        if not solution_path.exists():
            raise ValueError(
                f"{solution_path} does not exist, check argument path {path}"
            )

        write_exercise(solution_path, path)
