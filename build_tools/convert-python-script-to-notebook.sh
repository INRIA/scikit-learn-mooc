#!/bin/bash
set -e

PYTHON_SCRIPT=$1
NOTEBOOK=$2
JUPYTER_KERNEL=$3

TMP_NOTEBOOK=${NOTEBOOK/.ipynb/_tmp.ipynb}
jupytext --to notebook --output $TMP_NOTEBOOK $PYTHON_SCRIPT
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 \
        --executepreprocessor.kernel_name=$JUPYTER_KERNEL --inplace $TMP_NOTEBOOK

# Only move the temporary notebook if all the previous command have been successful
mv $TMP_NOTEBOOK $NOTEBOOK
