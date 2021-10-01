# %%
from pathlib import Path

root_dir = Path(__file__).parents[1]
# %%

import yaml
from pathlib import Path

from sphinx_external_toc.parsing import parse_toc_yaml

toc_path = Path("~/dev/scikit-learn-mooc/jupyter-book/_toc.yml").expanduser()
site_map = parse_toc_yaml(toc_path)

# %%
# It seems simpler to use the json
json_info = site_map.as_json()
json_info.keys()
# %%
documents = json_info['documents']
root_doc = documents[json_info['root']]
root_doc

# This looks like this has only the next level of information (lesson and not individual notebooks)
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
# dict has a 'items' key with only the first level (typically the _index file)
root_doc['subtrees'][:3]

# %%
# You can access more info of each document directly by name. For example the
# _index files will have a 'subtrees' key with only one element.
documents['predictive_modeling_pipeline/01_tabular_data_exploration_index']

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
# Some documents don't have subtrees, those are the ones where the lesson has no subparts
documents['predictive_modeling_pipeline/predictive_modeling_module_intro']

# %%
# Alternatively you can use the SiteMap object directly but this seems to have
# the same info as the json
doc = site_map.root
doc.subtrees
site_map[site_map.root.docname]
# %%
module, = [
    module for module in root_doc['subtrees'] if module['caption'] == 'The predictive modeling pipeline']

module

# %%
lessons = module['items']
# without subtrees (predictive modeling intro in this case)
documents[lessons[0]]
# %%
# with subtrees (tabular data exploration lesson index)
documents[lessons[1]]

# %%
from myst_parser.main import to_tokens

import jupytext

def get_first_title(path):
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist')
    if path.suffix == '.py':
        cells = jupytext.reads(path.read_text(), fmt='py:percent')
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

filepaths = [
    mooc_dir / 'python_scripts/01_tabular_data_exploration.py',
    mooc_dir / 'jupyter-book/predictive_modeling_pipeline/predictive_modeling_module_intro.md'
]

for path in filepaths:
    print(get_first_title(path))
# %%

def docname_to_path(docname):
    path_without_suffix = root_dir / 'jupyter-book' / docname
    for suffix in ['.py', '.md']:
        path = path_without_suffix.with_suffix(suffix)
        if path.exists():
            return path
    else:
        raise ValueError(f'No filename found for docname: {docname}')

def get_single_file_markdown(docname):
    """
    docname is the name present in _toc.yml i.e. path without suffix relative to
    jupyter-book
    """
    path = docname_to_path(docname)
    title = get_first_title(path)
    target = path
    # python_scripts is used in _toc.yml but the index needs to point to notebooks 
    if path.suffix == '.py':
        target = Path(str(path).replace('jupyter-book/', '').replace('python_scripts', 'notebooks').replace('.py', '.ipynb'))
    # for now the target is relative to the root_dir since that is where
    # index.md lives. Later this can be another argument of the script ?
    target = target.relative_to(root_dir)

    return f"[{title}]({target})"

for docname in [
    'python_scripts/01_tabular_data_exploration',
    'predictive_modeling_pipeline/predictive_modeling_module_intro' 
]:
    print(get_single_file_markdown(docname))
# get_root_doc_markdown(json)

# %%

def get_lesson_markdown(lesson):
    # lesson with no subparts vs lesson with subparts
    if not lesson['subtrees']:
        lesson_md = get_single_file_markdown(lesson['docname'])
        return lesson_md

    path = docname_to_path(lesson['docname'])
    lesson_title = get_first_title(path)
    heading = f"## {lesson_title}"

    lesson_docnames = lesson['subtrees'][0]['items']
    lesson_content = '\n'.join(f'* {get_single_file_markdown(docname)}'
        for docname in lesson_docnames)
    return f"{heading}\n\n{lesson_content}"

print(
    get_lesson_markdown(
        documents['predictive_modeling_pipeline/01_tabular_data_exploration_index']))

# %%
def get_module_markdown(module_dict, documents):
    module_title = module_dict['caption']
    heading = f"# {module_title}"
    content = "\n\n".join(get_lesson_markdown(documents[docname]) for docname in module_dict['items'])
    return f"{heading}\n\n{content}"

print(get_module_markdown({'items': ['predictive_modeling_pipeline/predictive_modeling_module_intro',
    'predictive_modeling_pipeline/01_tabular_data_exploration_index',
    'predictive_modeling_pipeline/02_numerical_pipeline_index',
    'predictive_modeling_pipeline/03_categorical_pipeline_index',
    'predictive_modeling_pipeline/wrap_up_quiz',
    'predictive_modeling_pipeline/predictive_modeling_module_take_away'],
    'caption': 'The predictive modeling pipeline'}, documents))
# %%
def get_markdown(json_info):
    documents = json_info['documents']
    root_doc = documents[json_info['root']]
    # TODO I need to remove some parts (index.md and wip modules)
    content = "\n\n".join(get_module_markdown(module, documents) for module in root_doc['subtrees'])
    return content

print(get_markdown(json_info))
# %%

# %%
