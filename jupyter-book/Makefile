PYTHON_SCRIPTS_DIR = ../python_scripts
RENDERED_NOTEBOOKS_DIR = rendered_notebooks
MD_FILES = $(shell ls $(PYTHON_SCRIPTS_DIR)/*.py | perl -pe "s@$(PYTHON_SCRIPTS_DIR)/(.+)\.py@\1.md@")

all: html

.PHONY: html

%.md: $(PYTHON_SCRIPTS_DIR)/%.py
	bash generate-md-file.sh $<

html: $(MD_FILES)
	jupyter-book build .

clean:
	jupyter-book clean .
