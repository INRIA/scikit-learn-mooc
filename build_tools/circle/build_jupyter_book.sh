#!/usr/bin/env bash
set -xe

function show_error_logs {
    echo "Some notebooks failed, see logs below:"
    for f in _build/html/reports/*.log; do
        echo "================================================================================"
        echo $f
        echo "================================================================================"
        cat $f
    done
    # You need to exit with non-zero here to cause the build to fail
    exit 1
}

apt-get install make

source /opt/conda/etc/profile.d/conda.sh
conda update --yes conda
conda create -n scikit-learn-mooc --yes python=3
conda activate scikit-learn-mooc
pip install -r requirements-dev.txt

cd jupyter-book
make 2>&1 | tee build.log

# Grep the log to make sure there has been no errors when running the notebooks
# since jupyter-book exit code is always 0
grep 'Execution Failed' build.log && show_error_logs || \
    echo 'All notebooks ran successfully'
