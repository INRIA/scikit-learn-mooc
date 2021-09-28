# %%

import yaml
from pathlib import Path

from sphinx_external_toc.parsing import parse_toc_yaml

toc_path = Path("~/dev/scikit-learn-mooc/jupyter-book/_toc.yml").expanduser()
site_map = parse_toc_yaml(toc_path)

# %%
# It seems simpler to use the json
json = site_map.as_json()
json.keys()
# %%
documents = json['documents']
root_doc = documents[json['root']]
root_doc

# %%
# It has 'subtrees' key which is a list of modules dict. Each module dict has a
# 'items' key with only the first level (typically the _index file)
root_doc['subtrees'][:3]

# %%
# You can access more info of each document directly by name. For example the
# _index files will have a 'subtrees' key with only one element.
documents['predictive_modeling_pipeline/01_tabular_data_exploration_index']

# %%
# Some documents don't have subtrees
documents['predictive_modeling_pipeline/predictive_modeling_module_intro']

# %%
# Alternatively you can use the SiteMap object directly but this seems to have the same info as the json
doc = site_map.root

# %%
doc.subtrees

# %%
site_map[site_map.root.docname]
# %%
module, = [
    module for module in root_doc['subtrees'] if module['caption'] == 'The predictive modeling pipeline']

module

# %%

lessons = module['items']
# without subtrees
documents[lessons[0]]
# %%
# with subtrees
documents[lessons[1]]

# %%
def get_root_doc_markdown(json):
    documents = json['documents']
    root_doc = documents[json['root']]
    module_md_list = [get_module_markdown(module, documents) for module in root_doc['subtrees']]
        

def get_module_markdown(module, documents):
    lesson_md_list = [get_lesson_markdown(lesson, documents) for lesson in module['items']]


def get_lesson_markdown(lesson, documents):
    # lesson with no subparts vs lesson with subparts
    single_document_md_list = [get_single_file_markdown(documents[name]) for name in documents[lesson]]

def get_single_file_markdown(doc):
    print(doc)

get_root_doc_markdown(json)
# %%
def_get_leaf_node_markdown(doc):
    pass

# %%
from myst_parser.main import to_tokens

import jupytext

def get_first_title(path):
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist')
    if path.suffix == '.py':
        cells = jupytext.reads(my_str, fmt='py:percent')
        md_str = jupytext.writes(cells, fmt='md:myst')
    elif path.suffix == '.md':
        md_str = path.read_text()
    else:
        raise ValueError(f'{filename} is not a .py or a .md file')

    return get_first_title_from_md_str(md_str)

def get_first_title_from_md_str(md_str):
    tokens = to_tokens(md_str)
    is_title_token = False
    for t in tokens:
        if is_title_token:
            title = t.children[0].content
            return title
        if t.type == 'heading_open':
            is_title_token = True

mooc_dir = Path('~/dev/scikit-learn-mooc').expanduser()
path = mooc_dir / 'python_scripts/01_tabular_data_exploration.py' 
get_first_title(path)
# %%
path = mooc_dir / 'jupyter-book/predictive_modeling_pipeline/predictive_modeling_module_intro.md'  
get_first_title(path)
# %%

# %%
