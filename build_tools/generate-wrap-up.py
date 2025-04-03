import sys
import os
import glob


def extract_python_code_blocks(md_file_path):
    """
    Extract Python code blocks from a markdown file.

    Args:
        md_file_path (str): Path to the markdown file

    Returns:
        list: List of extracted Python code blocks
    """
    code_blocks = []
    in_python_block = False
    current_block = []

    with open(md_file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")

            if line.strip() == "```python":
                in_python_block = True
                current_block = []
            elif line.strip() == "```" and in_python_block:
                in_python_block = False
                code_blocks.append("\n".join(current_block))
            elif in_python_block:
                current_block.append(line)

    return code_blocks


def write_jupyter_notebook_file(
    code_blocks, output_file="notebook_from_md.py"
):
    """
    Writes extracted code blocks to a Python file formatted as Jupyter notebook cells.

    Args:
        code_blocks (list): List of code blocks to write
        output_file (str): Path to the output file
    """
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(
            "# %% [markdown] \n # ## Notebook generated from Markdown file\n\n"
        )

        for i, block in enumerate(code_blocks, 1):
            file.write(f"# %% [markdown]\n# ## Cell {i}\n\n# %%\n{block}\n\n")

        print(
            f"Successfully wrote {len(code_blocks)} code cells to"
            f" {output_file}"
        )


def process_quiz_files(input_path, output_dir):
    """
    Process all wrap_up_quiz files in the input path and convert them to notebooks.

    Args:
        input_path (str): Path to look for wrap_up_quiz files in subfolders
        output_dir (str): Directory to write the generated notebooks
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find all files containing "wrap_up_quiz" in their name in the input path subfolders
    quiz_files = glob.glob(
        f"{input_path}/**/*wrap_up_quiz*.md", recursive=True
    )

    if not quiz_files:
        print(f"No wrap_up_quiz.md files found in {input_path} subfolders.")
        return

    print(f"Found {len(quiz_files)} wrap_up_quiz files to process.")

    # Process each file
    for md_file_path in quiz_files:
        print(f"\nProcessing: {md_file_path}")

        # Extract code blocks
        code_blocks = extract_python_code_blocks(md_file_path)

        # Generate output filename
        subfolder = md_file_path.split(os.sep)[3]  # Get subfolder name
        output_file = os.path.join(output_dir, f"{subfolder}_wrap_up_quiz.py")

        # Display results and write notebook file
        if code_blocks:
            print(f"Found {len(code_blocks)} Python code blocks")
            write_jupyter_notebook_file(code_blocks, output_file=output_file)
        else:
            print(f"No Python code blocks found in {md_file_path}.")


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    process_quiz_files(input_path, output_dir)
