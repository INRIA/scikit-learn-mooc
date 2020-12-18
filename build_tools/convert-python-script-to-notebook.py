import sys

from docutils import nodes
from docutils import utils
from docutils.core import publish_from_doctree

import nbformat

from bs4 import BeautifulSoup

from myst_parser.main import to_docutils

import jupytext

# https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#directives
# Docutils supports the following directives:
# Admonitions: attention, caution, danger, error, hint, important, note, tip,
# warning and the generic admonition. (Most themes style only “note” and
# “warning” specially.)
possible_admonitions = [
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

# follows JupterBook CSS
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


def convert_to_html(node, css_selector=None):
    new_doc = utils.new_document("notset")
    new_doc.append(node.deepcopy())
    html_str = publish_from_doctree(
        new_doc, writer_name="html").decode("utf-8")
    html_node = BeautifulSoup(html_str, features="html.parser")

    if css_selector is not None:
        html_node = html_node.select_one(css_selector)
    return html_node


def replace_admonition_with_html_node(node):
    html_node = convert_to_html(node, "div.admonition")
    bootstrap_class = sphinx_name_to_bootstrap[node.tagname]
    html_node.attrs["class"] += [f"alert alert-{bootstrap_class}"]
    html_node.select_one(
        ".admonition-title").attrs["style"] = "font-weight: bold;"
    replacement = nodes.raw(text=str(html_node), format="html")
    node.replace_self(replacement)


def replace_admonition_in_cell_source(cell_str):
    doc = to_docutils(cell_str)
    for adm in doc.traverse(nodes.Admonition):
        replace_admonition_with_html_node(adm)

    return doc.astext()


def replace_admonition_in_nb(nb):
    # FIXME this would not work with advanced syntax for admonition with :::
    # but we are not using it for now. We could parse all the markdowns cell, a
    # bit wasteful, but probably good enough
    cells_with_admonition = [
        (i, c)
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "markdown"
        and any(("{" + adm + "}") in c["source"]
                for adm in possible_admonitions)
    ]

    for i, c in cells_with_admonition:
        cell_src = c["source"]
        output_src = replace_admonition_in_cell_source(cell_src)
        nb.cells[i]["source"] = output_src


def replace_admonition_in_filename(input_filename, output_filename):
    nb = jupytext.read(input_filename)

    replace_admonition_in_nb(nb)

    with open(output_filename, "w") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    replace_admonition_in_filename(input_filename, output_filename)
