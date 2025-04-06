# %%
from pathlib import Path

import click

import nbformat

import jupytext

from sphinx_external_toc.parsing import parse_toc_yaml

from markdown_it.renderer import RendererHTML

from myst_parser.config.main import MdParserConfig
from myst_parser.parsers.mdit import create_md_parser


# This hard-code the git repo root directory relative to this script
root_dir = Path(__file__).parents[1]


def get_first_title_from_md_str(md_str):
    parser = create_md_parser(MdParserConfig(), RendererHTML)
    tokens = parser.parse(md_str)

    is_title_token = False
    for t in tokens:
        if is_title_token:
            title = t.children[0].content
            return title
        if t.type == "heading_open":
            is_title_token = True


def get_first_title(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path.suffix == ".py":
        cells = jupytext.reads(path.read_text(), fmt="py:percent")
        md_str = jupytext.writes(cells, fmt="md:myst")
    elif path.suffix == ".md":
        md_str = path.read_text()
    else:
        raise ValueError(f"{path} is not a .py or a .md file")

    return get_first_title_from_md_str(md_str)


def test_get_first_title():
    filepaths = [
        root_dir / "python_scripts/01_tabular_data_exploration.py",
        root_dir
        / "jupyter-book/predictive_modeling_pipeline/predictive_modeling_module_intro.md",
    ]

    for path in filepaths:
        print(get_first_title(path))


def docname_to_path(docname):
    path_without_suffix = root_dir / "jupyter-book" / docname
    for suffix in [".py", ".md"]:
        path = path_without_suffix.with_suffix(suffix)
        if path.exists():
            return path
    else:
        raise ValueError(f"No filename found for docname: {docname}")


def get_single_file_markdown(docname):
    """Returns markdown link from docname.

    docname is the name present in _toc.yml i.e. path without suffix relative to
    the jupyter-book folder.

    The title of the link is the first title from the docname. The target to the
    link point to a notebook or a markdown file.
    """
    path = docname_to_path(docname)
    title = get_first_title(path)
    target = path
    # For now the target is relative to the repo root directory since that is
    # where full-index.ipynb lives. Maybe one day this can be another argument of
    # the script ?
    target = target.relative_to(root_dir)

    if path.suffix == ".py":
        # python_scripts is used in _toc.yml but the index needs to point to notebooks
        target = Path(
            str(target)
            .replace("jupyter-book/", "")
            .replace("python_scripts", "notebooks")
            .replace(".py", ".ipynb")
        )
    elif path.suffix == ".md":
        # This is simpler to point to inria.github.io generated HTML otherwise
        # there are quirks (MyST in quizzes not supported, slides not working,
        # etc ...)
        relative_url = (
            str(target).replace("jupyter-book/", "").replace(".md", ".html")
        )
        target = f"https://inria.github.io/scikit-learn-mooc/{relative_url}"

    return f"[{title}]({target})"


def test_get_single_file_markdown():
    for docname in [
        "python_scripts/01_tabular_data_exploration",
        "predictive_modeling_pipeline/predictive_modeling_module_intro",
    ]:
        print(get_single_file_markdown(docname))


def get_lesson_markdown(lesson):
    """Return markdown for a full lesson.

    An example of lesson is the Tabular data exploration lesson in the
    Predictive modeling pipeline module.
    """
    if not lesson["subtrees"]:
        lesson_md = get_single_file_markdown(lesson["docname"])
        return lesson_md

    path = docname_to_path(lesson["docname"])
    lesson_title = get_first_title(path)
    # Use third (rather than second) level header to see more clearly the
    # difference between modules anda lesson
    heading = f"### {lesson_title}"

    lesson_docnames = lesson["subtrees"][0]["items"]
    lesson_content = "\n".join(
        f"* {get_single_file_markdown(docname)}" for docname in lesson_docnames
    )
    return f"{heading}\n\n{lesson_content}"


def test_get_lesson_markdown():
    toc_path = root_dir / "jupyter-book/_toc.yml"
    site_map = parse_toc_yaml(toc_path)
    json_info = site_map.as_json()
    documents = json_info["documents"]
    print(
        get_lesson_markdown(
            documents[
                "predictive_modeling_pipeline/01_tabular_data_exploration_index"
            ]
        )
    )


def get_module_markdown(module_dict, documents):
    """Return markdown for a full module.

    An example of module is the Predictive modeling pipeline module. module_dict
    has two important entries:
    - 'items': list of lesson docnames
    - 'caption': module title
    """
    module_title = module_dict["caption"]
    heading = f"# {module_title}"
    content = "\n\n".join(
        get_lesson_markdown(documents[docname])
        for docname in module_dict["items"]
    )
    return f"{heading}\n\n{content}"


def test_get_module_markdown():
    toc_path = Path("jupyter-book/_toc.yml")
    site_map = parse_toc_yaml(toc_path)
    json_info = site_map.as_json()
    documents = json_info["documents"]
    print(
        get_module_markdown(
            {
                "items": [
                    "predictive_modeling_pipeline/predictive_modeling_module_intro",
                    "predictive_modeling_pipeline/01_tabular_data_exploration_index",
                    "predictive_modeling_pipeline/02_numerical_pipeline_index",
                    "predictive_modeling_pipeline/03_categorical_pipeline_index",
                    "predictive_modeling_pipeline/wrap_up_quiz",
                    "predictive_modeling_pipeline/predictive_modeling_module_take_away",
                ],
                "caption": "The predictive modeling pipeline",
            },
            documents,
        )
    )


def get_full_index_markdown(toc_path):
    site_map = parse_toc_yaml(toc_path)
    json_info = site_map.as_json()

    documents = json_info["documents"]
    root_doc = documents[json_info["root"]]

    def should_keep_module(module_title):
        # Not sure exactly why but index is listed and has a None caption ...
        is_index = module_title is None
        if is_index:
            return False
        is_wip = "🚧" in module_title
        is_appendix = "Appendix" in module_title

        return not is_wip and not is_appendix

    content = "\n\n".join(
        get_module_markdown(module, documents)
        for module in root_doc["subtrees"]
        if should_keep_module(module["caption"])
    )
    return content


def test_get_full_index_markdown():
    toc_path = root_dir / "jupyter-book/_toc.yml"

    print(get_full_index_markdown(toc_path))


def get_full_index_ipynb(toc_path):
    md_str = get_full_index_markdown(toc_path)
    nb = jupytext.reads(md_str, format=".md")

    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(md_str)]
    )

    # nb_content = jupytext.writes(nb, fmt=".ipynb")
    # nb = json.loads(nb_content)

    # In nbformat 5, markdown cells have ids, nbformat.write and consequently
    # jupytext writes random cell ids when generating .ipynb from .py, creating
    # unnecessary changes.
    for c in nb.cells:
        del c["id"]

    return nbformat.v4.writes(nb)


def test_json_manipulation():
    """Gives a few hints about the json format"""
    # %%
    toc_path = root_dir / "jupyter-book/_toc.yml"
    site_map = parse_toc_yaml(toc_path)
    json_info = site_map.as_json()

    documents = json_info["documents"]
    root_doc = documents[json_info["root"]]
    root_doc

    # This looks like this has only the first level of information (lesson and
    # not individual notebooks)
    # {'docname': 'toc',
    #  'subtrees': [{'items': ['index'],
    #   'caption': None,
    #   ...},
    #   {'items': ['ml_concepts/slides', 'ml_concepts/quiz_intro_01'],
    #    'caption': 'Machine Learning Concepts',
    #    ...}
    #   {'items': ['predictive_modeling_pipeline/predictive_modeling_module_intro',
    #     'predictive_modeling_pipeline/01_tabular_data_exploration_index',
    #     'predictive_modeling_pipeline/02_numerical_pipeline_index',
    #     'predictive_modeling_pipeline/03_categorical_pipeline_index',
    #     'predictive_modeling_pipeline/wrap_up_quiz',
    #     'predictive_modeling_pipeline/predictive_modeling_module_take_away'],
    #    'caption': 'The predictive modeling pipeline',
    #    ...},
    # }

    # %%
    # The root doc has 'subtrees' key which is a list of modules dict. Each module
    # dict has a 'items' key with only the first level (typically the *_index files)
    root_doc["subtrees"][:3]

    # %%
    # You can access more info of each document directly by name. For example the
    # _index files will have a 'subtrees' key with only one element.
    documents["predictive_modeling_pipeline/01_tabular_data_exploration_index"]

    # So for a lesson index you need to access subtrees to have the individual notebooks
    # {'docname': 'predictive_modeling_pipeline/01_tabular_data_exploration_index',
    # 'title': None,
    #  'subtrees': [{'items': ['python_scripts/01_tabular_data_exploration',
    #     'python_scripts/01_tabular_data_exploration_ex_01',
    #     'python_scripts/01_tabular_data_exploration_sol_01',
    #     'predictive_modeling_pipeline/01_tabular_data_exploration_quiz_m1_01'],
    #    'caption': None,
    #    'hidden': True,
    #    'maxdepth': -1,
    #    'numbered': False,
    #    'reversed': False,
    #    'titlesonly': True}]}

    # %%
    (module,) = [
        module
        for module in root_doc["subtrees"]
        if module["caption"] == "The predictive modeling pipeline"
    ]

    module

    # %%
    lessons = module["items"]
    # without subtrees (predictive modeling intro in this case)
    documents[lessons[0]]
    # %%
    # with subtrees (tabular data exploration lesson index)
    documents[lessons[1]]


@click.command()
@click.option(
    "--toc",
    default="jupyter-book/_toc.yml",
    help="Path of the _toc.yml used for jupyter-book",
)
@click.option(
    "--output",
    default="full-index.ipynb",
    help="Path where the index notebook will be written",
)
def main(toc, output):
    toc_path = Path(toc)
    output_path = Path(output)

    ipynb_str = get_full_index_ipynb(toc_path)
    output_path.write_text(ipynb_str)


if __name__ == "__main__":
    main()
