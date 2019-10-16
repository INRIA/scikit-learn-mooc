PYTHON_SCRIPTS_DIR = python_scripts
NOTEBOOKS_DIR = notebooks
RENDERED_NOTEBOOKS_DIR = rendered_notebooks
JUPYTER_KERNEL := python3
MINIMAL_NOTEBOOK_FILES = $(shell ls $(PYTHON_SCRIPTS_DIR)/*.py | perl -pe "s@$(PYTHON_SCRIPTS_DIR)@$(NOTEBOOKS_DIR)@" | perl -pe "s@\.py@.ipynb@")
MINIMAL_RENDERED_NOTEBOOK_FILES = $(shell ls $(PYTHON_SCRIPTS_DIR)/*.py | perl -pe "s@$(PYTHON_SCRIPTS_DIR)@$(RENDERED_NOTEBOOKS_DIR)@" | perl -pe "s@\.py@.ipynb@")

all: $(RENDERED_NOTEBOOKS_DIR)

.PHONY: $(NOTEBOOKS_DIR) $(RENDERED_NOTEBOOKS_DIR) sanity_check_$(PYTHON_SCRIPTS_DIR) sanity_check_$(NOTEBOOKS_DIR) sanity_check_$(RENDERED_NOTEBOOKS_DIR) all

$(NOTEBOOKS_DIR): sanity_check_$(PYTHON_SCRIPTS_DIR) $(MINIMAL_NOTEBOOK_FILES) sanity_check_$(NOTEBOOKS_DIR)

$(RENDERED_NOTEBOOKS_DIR): $(MINIMAL_RENDERED_NOTEBOOK_FILES) sanity_check_$(RENDERED_NOTEBOOKS_DIR)

$(NOTEBOOKS_DIR)/%.ipynb: $(PYTHON_SCRIPTS_DIR)/%.py
	jupytext --set-formats $(PYTHON_SCRIPTS_DIR)//py:percent,$(NOTEBOOKS_DIR)//ipynb $<
	jupytext --sync $<

$(RENDERED_NOTEBOOKS_DIR)/%.ipynb: $(NOTEBOOKS_DIR)/%.ipynb
	cp $< $@
	jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=None \
                    --ExecutePreprocessor.kernel_name=$(JUPYTER_KERNEL) --inplace $@

sanity_check_$(PYTHON_SCRIPTS_DIR):
	python build_scripts/check-python-scripts.py $(PYTHON_SCRIPTS_DIR)
	yapf --recursive --in-place --parallel $(PYTHON_SCRIPTS_DIR)

sanity_check_$(NOTEBOOKS_DIR):
	python build_scripts/sanity-check.py $(PYTHON_SCRIPTS_DIR) $(NOTEBOOKS_DIR)

sanity_check_$(RENDERED_NOTEBOOKS_DIR):
	python build_scripts/sanity-check.py $(NOTEBOOKS_DIR) $(RENDERED_NOTEBOOKS_DIR)
