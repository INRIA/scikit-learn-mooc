import sys

from docutils.core import publish_from_doctree

from bs4 import BeautifulSoup

from myst_parser.main import MdParserConfig, default_parser

import jupytext

# https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#directives
# Docutils supports the following directives:
# Admonitions: attention, caution, danger, error, hint, important, note, tip,
# warning and the generic admonition
all_admonitions = [
    "attention",
    "caution",
    "danger",
    "error",
    "hint",
    "important",
    "note",
    "tip",
    "warning",
    "admonition",
]

# follows colors defined by JupterBook CSS
sphinx_name_to_bootstrap = {
    "attention": "warning",
    "caution": "warning",
    "danger": "danger",
    "error": "danger",
    "hint": "warning",
    "important": "info",
    "note": "info",
    "tip": "warning",
    "warning": "danger",
    "admonition": "info",
}

all_directive_names = ["{" + adm + "}" for adm in all_admonitions]


def convert_to_html(doc, css_selector=None):
    """Converts docutils document to HTML and select part of it with CSS
    selector.
    """
    html_str = publish_from_doctree(doc, writer_name="html").decode("utf-8")
    html_node = BeautifulSoup(html_str, features="html.parser")

    if css_selector is not None:
        html_node = html_node.select_one(css_selector)

    return html_node


def admonition_html(doc):
    """Returns admonition HTML from docutils document.

    Assumes that the docutils document has a single child which is an
    admonition.
    """
    assert len(doc.children) == 1
    adm_node = doc.children[0]
    assert adm_node.tagname in all_admonitions
    html_node = convert_to_html(doc, "div.admonition")
    bootstrap_class = sphinx_name_to_bootstrap[adm_node.tagname]
    html_node.attrs["class"] += [f"alert alert-{bootstrap_class}"]
    html_node.select_one(
        ".admonition-title").attrs["style"] = "font-weight: bold;"

    return str(html_node)


def replace_admonition_in_cell_source(cell_str):
    """Returns cell source with admonition replaced by its generated HTML.
    """
    config = MdParserConfig(renderer="docutils")
    parser = default_parser(config)
    tokens = parser.parse(cell_str)

    admonition_tokens = [
        t for t in tokens
        if t.type == "fence" and t.info in all_directive_names
    ]

    cell_lines = cell_str.splitlines()
    new_cell_str = cell_str

    for t in admonition_tokens:
        adm_begin, adm_end = t.map
        adm_src = "\n".join(cell_lines[adm_begin:adm_end])
        adm_doc = parser.render(adm_src)
        adm_html = admonition_html(adm_doc)
        new_cell_str = new_cell_str.replace(adm_src, adm_html)

    return new_cell_str


def replace_admonition_in_nb(nb):
    """Replaces all admonitions by its generated HTML in a notebook object.
    """
    # FIXME this would not work with advanced syntax for admonition with :::
    # but we are not using it for now. We could parse all the markdowns cell, a
    # bit wasteful, but probably good enough
    cells_with_admonition = [
        (i, c)
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "markdown"
        and any(directive in c["source"] for directive in all_directive_names)
    ]

    for i, c in cells_with_admonition:
        cell_src = c["source"]
        output_src = replace_admonition_in_cell_source(cell_src)
        nb.cells[i]["source"] = output_src


def replace_admonition_in_filename(input_filename, output_filename):
    """Converts .py file to .ipynb file where admonitions have been replaced by
    their generated HTML.

    Context: MyST syntax is not supported inside a Jupyter notebook. This is a
    hacky way to keep using MyST admonitions for our JupyterBook and have
    acceptable admonition HTML in the Jupyter notebook interface.
    """
    nb = jupytext.read(input_filename)

    replace_admonition_in_nb(nb)

    jupytext.write(nb, output_filename)


if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    replace_admonition_in_filename(input_filename, output_filename)
