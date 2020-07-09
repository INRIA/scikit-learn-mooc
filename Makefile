PYTHON_SCRIPTS_DIR = python_scripts
RENDERED_NOTEBOOKS_DIR = rendered_notebooks
JUPYTER_KERNEL := python3
MINIMAL_RENDERED_NOTEBOOK_FILES = $(shell ls $(PYTHON_SCRIPTS_DIR)/*.py | perl -pe "s@$(PYTHON_SCRIPTS_DIR)@$(RENDERED_NOTEBOOKS_DIR)@" | perl -pe "s@\.py@.ipynb@")

all: $(RENDERED_NOTEBOOKS_DIR)

.PHONY: $(RENDERED_NOTEBOOKS_DIR) sanity_check_$(PYTHON_SCRIPTS_DIR) sanity_check_$(RENDERED_NOTEBOOKS_DIR) all clean

$(RENDERED_NOTEBOOKS_DIR): $(MINIMAL_RENDERED_NOTEBOOK_FILES) sanity_check_$(RENDERED_NOTEBOOKS_DIR)

$(RENDERED_NOTEBOOKS_DIR)/%.ipynb: $(PYTHON_SCRIPTS_DIR)/%.py
	bash build_tools/convert-python-script-to-notebook.sh $< $@ $(JUPYTER_KERNEL)

sanity_check_$(PYTHON_SCRIPTS_DIR):
	python build_tools/check-python-scripts.py $(PYTHON_SCRIPTS_DIR)
	yapf --recursive --in-place --parallel $(PYTHON_SCRIPTS_DIR)

sanity_check_$(RENDERED_NOTEBOOKS_DIR):
	python build_tools/sanity-check.py $(PYTHON_SCRIPTS_DIR) $(RENDERED_NOTEBOOKS_DIR)

clean:
	rm -rf $(RENDERED_NOTEBOOKS_DIR)/*tmp.ipynb
