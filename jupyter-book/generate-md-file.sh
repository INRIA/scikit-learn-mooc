#!/bin/bash

set -e

py_filename=$1
md_filename=$(basename $py_filename .py).md

jupytext --to myst $py_filename --output $md_filename
