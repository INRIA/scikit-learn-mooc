#!/usr/bin/env bash
set -xe

apt-get install make

source /opt/conda/etc/profile.d/conda.sh
conda update --yes conda
conda create -n testenv --yes pip python=3.7
conda activate testenv
pip install -r requirements.txt
pip install jupytext yapf
pip install jupyter-book

cd jupyter-book
jupyter-book build .
