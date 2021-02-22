#!/usr/bin/env bash
set -xe

apt-get install make

source /opt/conda/etc/profile.d/conda.sh
conda update --yes conda
conda create -n scikit-learn-mooc --yes python=3
conda activate scikit-learn-moooc
pip install -r requirements-dev.txt

cd jupyter-book
make 2>&1 | tee build.log

# Grep the log to make sure there has been no errors when running the notebooks
# since jupyter-book exit code is always 0
grep 'Execution Failed' build.log && exit 1 || \
    echo 'All notebooks ran successfully'
