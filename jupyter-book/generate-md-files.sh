#!/bin/bash

set -e

for f in ../python_scripts/0[1-4]*.py; do
    jupytext --to myst $f --output $(basename $f .py).md
done

for f in 0[1-4]*.md; do
    jupytext --set-kernel scikit-learn-tutorial $f
done
