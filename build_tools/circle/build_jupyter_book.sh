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

affected_jupyter_book_paths() {
    files=$(git diff --name-only origin/master...$CIRCLE_SHA1)
    # TODO: rather than the grep pattern below we could potentially look at
    # _toc.yml to know whether the file affects the JupyterBook
    echo "$files" | grep python_scripts | perl -pe 's@\.py$@.html@'
    echo "$files" | grep -P 'jupyter-book/.+md$' | perl -pe 's@jupyter-book/(.+)\.md@\1.html@'
}

write_changed_html() {
    affected="$1"
    if [ -n "$CI_PULL_REQUEST" ]
    then
        echo "The following files may have been changed by PR #$CI_PULL_REQUEST:"
        echo "$affected"
        (
            echo '<html><body>'
            echo 'Files changed by PR <a href="'"$CI_PULL_REQUEST"'">'"$CI_PULL_REQUEST</a>"
            echo '<ul>'
            echo "$affected" | sed 's|.*|<li><a href="&">&</a> [<a href="https://inria.github.io/scikit-learn-mooc/&">master</a>]|'
            echo '</ul><p>This PR JupyterBook <a href="index.html">index</a>'
            echo '</ul></body></html>'
        ) > '_build/html/_changed.html'
    fi
}

affected=$(affected_jupyter_book_paths)

cd jupyter-book
make 2>&1 | tee build.log

write_changed_html "$affected"

# Grep the log to make sure there has been no errors when running the notebooks
# since jupyter-book exit code is always 0
grep 'Execution Failed' build.log && show_error_logs || \
    echo 'All notebooks ran successfully'
