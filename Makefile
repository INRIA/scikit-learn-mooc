PYTHON_SCRIPTS_DIR = python_scripts
NOTEBOOKS_DIR = notebooks
JUPYTER_BOOK_DIR = jupyter-book
WRAP_UP_DIR = wrap-up
JUPYTERLITE_DIR = jupyterlite
JUPYTER_KERNEL := python3
MINIMAL_NOTEBOOK_FILES = $(shell ls $(PYTHON_SCRIPTS_DIR)/*.py | perl -pe "s@$(PYTHON_SCRIPTS_DIR)@$(NOTEBOOKS_DIR)@" | perl -pe "s@\.py@.ipynb@")

# This assumes that the folder mooc-scikit-learn-coordination and
# scikit-learn-mooc are siblings, e.g. the repos are in the
# ~/dev/mooc-scikit-learn-coordination and ~/dev/scikit-learn-mooc. This should
# be the case in most development setups. If not then you can pass the
# GITLAB_REPO_JUPYTERBOOK_DIR variable with
# make -e GITLAB_REPO_JUPYTERBOOK_DIR=your/gitlab/repo/jupyter-book-dir/goes-here
GITLAB_REPO_JUPYTERBOOK_DIR = ../mooc-scikit-learn-coordination/jupyter-book

all: $(NOTEBOOKS_DIR)

.PHONY: $(NOTEBOOKS_DIR) copy_matplotlibrc sanity_check_$(NOTEBOOKS_DIR) all \
        exercises quizzes $(JUPYTER_BOOK_DIR) $(JUPYTER_BOOK_DIR)-clean $(JUPYTER_BOOK_DIR)-full-clean \
        $(JUPYTERLITE_DIR) $(JUPYTERLITE_DIR)-clean

$(NOTEBOOKS_DIR): $(MINIMAL_NOTEBOOK_FILES) copy_matplotlibrc sanity_check_$(NOTEBOOKS_DIR)

$(NOTEBOOKS_DIR)/%.ipynb: $(PYTHON_SCRIPTS_DIR)/%.py
	python build_tools/convert-python-script-to-notebook.py $< $@

copy_matplotlibrc:
	cp $(PYTHON_SCRIPTS_DIR)/matplotlibrc $(NOTEBOOKS_DIR)/

sanity_check_$(NOTEBOOKS_DIR):
	python build_tools/sanity-check.py $(PYTHON_SCRIPTS_DIR) $(NOTEBOOKS_DIR)

exercises:
	python build_tools/generate-exercise-from-solution.py $(PYTHON_SCRIPTS_DIR)

quizzes:
	python build_tools/generate-quizzes.py $(GITLAB_REPO_JUPYTERBOOK_DIR) $(JUPYTER_BOOK_DIR)

full-index:
	python build_tools/generate-index.py

run-code-in-wrap-up-quizzes:
	python build_tools/generate-wrap-up.py $(GITLAB_REPO_JUPYTERBOOK_DIR) $(WRAP_UP_DIR)
	jupytext --execute --to notebook $(WRAP_UP_DIR)/*.py

$(JUPYTER_BOOK_DIR):
	jupyter-book build $(JUPYTER_BOOK_DIR)
	rm -rf $(JUPYTER_BOOK_DIR)/_build/html/{slides,figures} && cp -r slides figures $(JUPYTER_BOOK_DIR)/_build/html

$(JUPYTER_BOOK_DIR)-clean:
	# keep jupyter-cache cache folder
	jupyter-book clean $(JUPYTER_BOOK_DIR)

$(JUPYTER_BOOK_DIR)-full-clean:
	# deletes jupyter-cache cache folder
	rm -rf $(JUPYTER_BOOK_DIR)/_build

# Build the JupyterLite site (xeus-python kernel) in jupyterlite/_output.
# Requires the build environment from jupyterlite/build-environment.yml.
# Notebooks and git-tracked datasets are staged side by side so that the
# relative "../datasets/..." paths used in the notebooks keep working.
# Always build from scratch: incremental builds keep stale kernel packages and
# skip re-applying jupyterlite/jupyter-lite.json (default kernel).
$(JUPYTERLITE_DIR): $(JUPYTERLITE_DIR)-clean
	mkdir -p $(JUPYTERLITE_DIR)/content/$(NOTEBOOKS_DIR)
	cp $(NOTEBOOKS_DIR)/*.ipynb $(NOTEBOOKS_DIR)/matplotlibrc $(JUPYTERLITE_DIR)/content/$(NOTEBOOKS_DIR)/
	git ls-files datasets | xargs cp --parents -t $(JUPYTERLITE_DIR)/content
	cd $(JUPYTERLITE_DIR) && jupyter lite build

$(JUPYTERLITE_DIR)-clean:
	rm -rf $(JUPYTERLITE_DIR)/content $(JUPYTERLITE_DIR)/_output $(JUPYTERLITE_DIR)/.jupyterlite.doit.db
