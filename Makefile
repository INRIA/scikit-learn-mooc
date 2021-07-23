PYTHON_SCRIPTS_DIR = python_scripts
NOTEBOOKS_DIR = notebooks
JUPYTER_KERNEL := python3
MINIMAL_NOTEBOOK_FILES = $(shell ls $(PYTHON_SCRIPTS_DIR)/*.py | perl -pe "s@$(PYTHON_SCRIPTS_DIR)@$(NOTEBOOKS_DIR)@" | perl -pe "s@\.py@.ipynb@")

all: $(NOTEBOOKS_DIR)

.PHONY: $(NOTEBOOKS_DIR) copy_matplotlibrc sanity_check_$(NOTEBOOKS_DIR) all

$(NOTEBOOKS_DIR): $(MINIMAL_NOTEBOOK_FILES) copy_matplotlibrc sanity_check_$(NOTEBOOKS_DIR)

$(NOTEBOOKS_DIR)/%.ipynb: $(PYTHON_SCRIPTS_DIR)/%.py
	python build_tools/convert-python-script-to-notebook.py $< $@

copy_matplotlibrc:
	cp $(PYTHON_SCRIPTS_DIR)/matplotlibrc $(NOTEBOOKS_DIR)/

sanity_check_$(NOTEBOOKS_DIR):
	python build_tools/sanity-check.py $(PYTHON_SCRIPTS_DIR) $(NOTEBOOKS_DIR)

exercises:
	python build_tools/generate-exercise-from-solution.py $(PYTHON_SCRIPTS_DIR)
