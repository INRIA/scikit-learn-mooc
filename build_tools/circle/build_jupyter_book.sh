#!/usr/bin/env bash
set -xe

apt-get install make

source /opt/conda/etc/profile.d/conda.sh
conda update --yes conda
conda create -n testenv --yes pip python=3.7
conda activate testenv
pip install -r requirements.txt
pip install jupyter-book
# nbformat 5.1 adds random id which creates problems with jupyter-cache
# https://github.com/mwouts/jupytext/issues/715
pip install nbformat==5.0.8

cd jupyter-book
make 2>&1 | tee build.log

# Grep the log to make sure there has been no errors when running the notebooks
# since jupyter-book exit code is always 0
grep 'Execution Failed' build.log && exit 1 || \
    echo 'All notebooks ran successfully'
